// src/app/api/shifts/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { pool } from '@/lib/db';
import { format } from 'date-fns';
import { User } from '@/types/shifts';

// Вспомогательные функции и GET/POST остаются без изменений
async function getUserFromSession(): Promise<User | null> {
  try {
    const cookie = cookies().get('auth-session');
    if (!cookie) return null;
    const parsed = JSON.parse(cookie.value);
    return parsed?.id ? parsed : null;
  } catch (err) {
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

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const viewedUserIdParam = url.searchParams.get('userId');
    const startParam = url.searchParams.get('start');
    const endParam = url.searchParams.get('end');

    const startStr = startParam ? normalizeDateString(startParam) : null;
    const endStr = endParam ? normalizeDateString(endParam) : null;

    const currentUser = await getUserFromSession();

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
    } else if (viewedUserIdParam) {
      params.push(parseInt(viewedUserIdParam, 10));
      where.push(`s.user_id = $${params.length}`);
    } else {
      where.push(`s.status = 'available'`);
    }

    if (where.length) {
      query += ' WHERE ' + where.join(' AND ');
    }
    query += ' ORDER BY s.shift_date, s.id';
    
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
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { date, startTime, endTime, assignToSelf } = await request.json();
    if (!date || !startTime || !endTime) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const dateStr = normalizeDateString(date);
    if (!dateStr) {
      return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
    }
    const shiftCode = `${startTime}-${endTime}`;
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      const existing = await client.query(
        'SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2',
        [dateStr, shiftCode]
      );

      if ((existing.rowCount ?? 0) > 0) {
        await client.query('COMMIT');
        return NextResponse.json(existing.rows[0]);
      }
      const userId = assignToSelf ? currentUser.id : null;
      const status = assignToSelf ? 'pending' : 'available';
      const insert = await client.query(
        `INSERT INTO shifts (shift_date, day_of_week, shift_code, status, user_id)
         VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, $3, $4)
         RETURNING *`,
        [dateStr, shiftCode, status, userId]
      );
      await client.query('COMMIT');
      return NextResponse.json(insert.rows[0], { status: 201 });

    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }
  } catch (err) {
    console.error('[POST /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// --- НАЧАЛО ИЗМЕНЕНИЙ ---
export async function DELETE(request: NextRequest) {
  const currentUser = await getUserFromSession();
  if (!currentUser) {
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
  
  // Используем транзакцию для безопасной очистки дня
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    
    console.log(`[API Shifts DELETE] User ${currentUser.id} starting to clear day ${dateStr}`);

    // Шаг 1: Освобождаем все слоты, занятые ТЕКУЩИМ пользователем в этот день.
    // Мы не трогаем слоты, занятые другими.
    const releaseResult = await client.query(
      `UPDATE shifts SET user_id = NULL, status = 'available' WHERE shift_date = $1 AND user_id = $2`,
      [dateStr, currentUser.id]
    );
    console.log(`[API Shifts DELETE] Released ${releaseResult.rowCount} slots taken by user ${currentUser.id}.`);

    // Шаг 2: Удаляем все СВОБОДНЫЕ слоты в этот день.
    // Это очищает "шаблоны", которые больше не нужны.
    const deleteResult = await client.query(
      `DELETE FROM shifts WHERE shift_date = $1 AND user_id IS NULL`,
      [dateStr]
    );
    console.log(`[API Shifts DELETE] Deleted ${deleteResult.rowCount} available slots.`);

    await client.query('COMMIT');

    return NextResponse.json({ 
      ok: true, 
      releasedCount: releaseResult.rowCount ?? 0,
      deletedCount: deleteResult.rowCount ?? 0,
    });

  } catch (err) {
    await client.query('ROLLBACK');
    console.error('[DELETE /api/shifts] transaction error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  } finally {
    client.release();
  }
}
// --- КОНЕЦ ИЗМЕНЕНИЙ ---