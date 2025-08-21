// src/app/api/shifts/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { pool } from '@/lib/db';
import { format } from 'date-fns';
import { User } from '@/types/shifts';

// --- Вспомогательные функции ---

async function getUserFromSession(): Promise<User | null> {
  try {
    const cookie = cookies().get('auth-session');
    if (!cookie) return null;
    const parsed = JSON.parse(cookie.value);
    return parsed?.id ? parsed : null;
  } catch (err) {
    console.error('[API Shifts] getUserFromSession error:', err);
    return null;
  }
}

function normalizeDateString(d: string | Date): string | null {
  try {
    const dt = typeof d === 'string' ? new Date(d) : d;
    if (Number.isNaN(dt.getTime())) return null;
    return format(dt, 'yyyy-MM-dd');
  } catch {
    return null;
  }
}

// --- Обработчики HTTP-методов ---

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const viewedUserIdParam = url.searchParams.get('userId');
    const startParam = url.searchParams.get('start');
    const endParam = url.searchParams.get('end');

    const startStr = startParam ? normalizeDateString(startParam) : null;
    const endStr = endParam ? normalizeDateString(endParam) : null;

    const currentUser = await getUserFromSession();
    
    console.log(`[API Shifts GET] currentUser: ${currentUser?.id}, viewedUserId: ${viewedUserIdParam}`);

    let query = `
      SELECT s.id, s.user_id, s.shift_date, s.shift_code, s.status,
             u.username, u.full_name
      FROM shifts s
      LEFT JOIN users u ON u.id = s.user_id
    `;
    const where: string[] = [];
    const params: any[] = [];

    if (startStr) {
      params.push(startStr);
      where.push(`s.shift_date >= $${params.length}`);
    }
    if (endStr) {
      params.push(endStr);
      where.push(`s.shift_date < $${params.length}`);
    }
    
    const isOwnerView =
      !viewedUserIdParam ||
      (currentUser && currentUser.id.toString() === viewedUserIdParam);

    if (isOwnerView && currentUser) {
      params.push(currentUser.id);
      where.push(`(s.user_id = $${params.length} OR s.status = 'available')`);
       console.log('[API Shifts GET] Viewing own schedule or all available slots.');
    } else if (viewedUserIdParam) {
      params.push(parseInt(viewedUserIdParam, 10));
      where.push(`s.user_id = $${params.length}`);
       console.log(`[API Shifts GET] Viewing schedule for user ID: ${viewedUserIdParam}.`);
    } else {
      where.push(`s.status = 'available'`);
       console.log('[API Shifts GET] Viewing only available slots (not logged in or no user specified).');
    }

    if (where.length) {
      query += ' WHERE ' + where.join(' AND ');
    }
    query += ' ORDER BY s.shift_date, s.id';
    
    console.log('[API Shifts GET] Executing query:', query, 'with params:', params);

    const result = await pool.query(query, params);
    return NextResponse.json(result.rows);
  } catch (err) {
    console.error('[GET /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) {
      console.warn('[API Shifts POST] Unauthorized attempt to create slot.');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { date, startTime, endTime, assignToSelf } = await request.json();
    console.log(`[API Shifts POST] User ${currentUser.id} creating slot for date: ${date} from ${startTime} to ${endTime}. Assign to self: ${assignToSelf}`);
    
    if (!date || !startTime || !endTime) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const dateStr = normalizeDateString(date);
    if (!dateStr) {
      return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
    }

    const shiftCode = `${startTime}-${endTime}`;
    
    // ВАЖНО: Транзакция, чтобы обеспечить атомарность операции
    const client = await pool.connect();
    try {
      await client.query('BEGIN');

      // Сначала ищем. Если нашли - просто возвращаем.
      const existing = await client.query(
        'SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2',
        [dateStr, shiftCode]
      );

      if ((existing.rowCount ?? 0) > 0) {
        console.log(`[API Shifts POST] Slot already exists for ${dateStr} ${shiftCode}. Returning existing.`);
        await client.query('COMMIT');
        return NextResponse.json(existing.rows[0]);
      }

      // Если не нашли - создаем.
      // --- НАЧАЛО ИЗМЕНЕНИЙ ---
      // Если `assignToSelf` true, сразу присваиваем слот текущему пользователю
      const userId = assignToSelf ? currentUser.id : null;
      const status = assignToSelf ? 'pending' : 'available';
      
      console.log(`[API Shifts POST] Creating new slot. Assigned User ID: ${userId}, Status: ${status}`);

      const insert = await client.query(
        `INSERT INTO shifts (shift_date, day_of_week, shift_code, status, user_id)
         VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, $3, $4)
         RETURNING *`,
        [dateStr, shiftCode, status, userId]
      );
      // --- КОНЕЦ ИЗМЕНЕНИЙ ---
      
      await client.query('COMMIT');
      console.log(`[API Shifts POST] Successfully created new slot with ID: ${insert.rows[0].id}`);
      return NextResponse.json(insert.rows[0], { status: 201 });

    } catch (err) {
      await client.query('ROLLBACK');
      throw err; // Передаем ошибку выше для обработки
    } finally {
      client.release();
    }

  } catch (err) {
    console.error('[POST /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) {
      console.warn('[API Shifts DELETE] Unauthorized attempt to delete slots.');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const url = new URL(request.url);
    const dateParam = url.searchParams.get('date');
    if (!dateParam) {
      return NextResponse.json({ error: 'Missing date param' }, { status: 400 });
    }

    const dateStr = normalizeDateString(dateParam);
    if (!dateStr) {
      return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
    }
    
    console.log(`[API Shifts DELETE] User ${currentUser.id} is releasing all slots for date: ${dateStr}`);
    
    const res = await pool.query(
      `UPDATE shifts SET user_id = NULL, status = 'available'
       WHERE shift_date = $1 AND user_id = $2`,
      [dateStr, currentUser.id]
    );
    console.log(`[API Shifts DELETE] Released ${res.rowCount} slots.`);
    return NextResponse.json({ ok: true, releasedCount: res.rowCount ?? 0 });
  } catch (err) {
    console.error('[DELETE /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}