// src/app/api/shifts/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { pool } from '@/lib/db';
import { format } from 'date-fns';
import { User } from '@/types/shifts';

/**
 * Универсальная функция для аутентификации пользователя.
 * Сначала проверяет API-ключ в заголовке, затем - сессионный cookie.
 * @param request - Входящий NextRequest.
 * @returns Promise<User | null> - Объект пользователя или null.
 */
async function getUserFromRequest(request: NextRequest): Promise<User | null> {
    // 1. Аутентификация по API-ключу (для ботов)
    const authHeader = request.headers.get('Authorization');
    if (authHeader && authHeader.startsWith('Bearer ')) {
        const apiKey = authHeader.substring(7);
        if (apiKey) {
            const { rows } = await pool.query<User>('SELECT id, username, full_name FROM users WHERE api_key = $1', [apiKey]);
            if (rows.length > 0) {
                return rows[0];
            }
        }
    }

    // 2. Аутентификация по cookie (для браузера)
    try {
        const cookie = cookies().get('auth-session');
        if (!cookie) return null;
        const parsed = JSON.parse(cookie.value);
        return parsed?.id ? parsed : null;
    } catch {
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

// Добавляем OPTIONS для CORS
export async function OPTIONS(request: NextRequest) {
    return new NextResponse(null, {
        status: 200,
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400',
        },
    });
}

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const viewedUserIdParam = url.searchParams.get('userId');
    const startParam = url.searchParams.get('start');
    const endParam = url.searchParams.get('end');

    const startStr = startParam ? normalizeDateString(startParam) : null;
    const endStr = endParam ? normalizeDateString(endParam) : null;

    const currentUser = await getUserFromRequest(request);

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
  console.log('[POST /api/shifts] ========== REQUEST START ==========');
  console.log('[POST /api/shifts] Method:', request.method);
  console.log('[POST /api/shifts] URL:', request.url);
  console.log('[POST /api/shifts] Headers:', Object.fromEntries(request.headers.entries()));
  
  try {
    // Проверяем тело запроса
    const body = await request.json();
    console.log('[POST /api/shifts] Body:', body);
    
    // Используем функцию аутентификации
    const currentUser = await getUserFromRequest(request);
    console.log('[POST /api/shifts] Current user:', currentUser);

    if (!currentUser) {
      console.log('[POST /api/shifts] Unauthorized - no user found');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { date, startTime, endTime, assignToSelf } = body;
    if (!date || !startTime || !endTime) {
      console.log('[POST /api/shifts] Missing fields:', { date, startTime, endTime });
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const dateStr = normalizeDateString(date);
    if (!dateStr) {
      return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
    }
    
    console.log('[POST /api/shifts] Processing shift:', { dateStr, startTime, endTime, assignToSelf });
    
    const shiftCode = `${startTime}-${endTime}`;
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      
      // ИСПРАВЛЕНИЕ: Проверяем существующий слот ТОЛЬКО для текущего пользователя
      const existing = await client.query(
        'SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2 AND user_id = $3',
        [dateStr, shiftCode, currentUser.id]
      );

      if ((existing.rowCount ?? 0) > 0) {
        // Пользователь уже имеет этот слот
        await client.query('COMMIT');
        console.log('[POST /api/shifts] User already has this shift:', existing.rows[0]);
        return NextResponse.json(existing.rows[0], { status: 200 });
      }
      
      // Слот не существует у этого пользователя - создаем новый
      const userId = assignToSelf ? currentUser.id : null;
      const status = assignToSelf ? 'pending' : 'available';
      
      const insert = await client.query(
        `INSERT INTO shifts (shift_date, day_of_week, shift_code, status, user_id)
         VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, $3, $4)
         RETURNING *`,
        [dateStr, shiftCode, status, userId]
      );
      
      await client.query('COMMIT');
      
      console.log('[POST /api/shifts] New shift created for user:', insert.rows[0]);
      return NextResponse.json(insert.rows[0], { status: 201 });

    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }
  } catch (err) {
    console.error('[POST /api/shifts] Error:', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  const currentUser = await getUserFromRequest(request);
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
  
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    
    console.log(`[API Shifts DELETE] User ${currentUser.id} clearing their shifts for ${dateStr}`);

    // ИСПРАВЛЕНИЕ: Удаляем только слоты ТЕКУЩЕГО пользователя
    const deleteResult = await client.query(
      `DELETE FROM shifts WHERE shift_date = $1 AND user_id = $2`,
      [dateStr, currentUser.id]
    );
    
    console.log(`[API Shifts DELETE] Deleted ${deleteResult.rowCount} shifts for user ${currentUser.id}.`);

    await client.query('COMMIT');

    return NextResponse.json({ 
      ok: true, 
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