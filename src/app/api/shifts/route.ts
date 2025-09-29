// src/app/api/shifts/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { pool } from '@/lib/db';
import { format } from 'date-fns';
import { User } from '@/types/shifts';

/**
 * Универсальная функция для аутентификации пользователя.
 * @param request - Входящий NextRequest.
 * @returns Promise<User | null> - Объект пользователя или null.
 */
async function getUserFromRequest(request: NextRequest): Promise<User | null> {
    const authHeader = request.headers.get('Authorization');
    if (authHeader && authHeader.startsWith('Bearer ')) {
        const apiKey = authHeader.substring(7);
        if (apiKey) {
            const { rows } = await pool.query<Omit<User, 'isOwner'>>('SELECT id, username, full_name FROM users WHERE api_key = $1', [apiKey]);
            if (rows.length > 0) {
                // Определяем isOwner по id пользователя (например, id=1 - владелец)
                return { ...rows[0], isOwner: rows[0].id === 1 };
            }
        }
    }

    try {
        const cookie = cookies().get('auth-session');
        if (!cookie) return null;
        const parsed = JSON.parse(cookie.value);
        if (parsed?.id) {
            return { ...parsed, isOwner: parsed.id === 1 };
        }
        return null;
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
    const allUsersParam = url.searchParams.get('allUsers'); // ДОБАВЛЕНО: параметр для получения всех слотов

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
    
    // ИЗМЕНЕНО: Добавлена проверка параметра allUsers
    if (allUsersParam === 'true') {
      // Если allUsers=true, показываем ВСЕ занятые слоты (не показываем свободные)
      where.push(`s.user_id IS NOT NULL`);
      console.log('[GET /api/shifts] Getting all users slots');
    } else {
      // Стандартная логика фильтрации
      if (viewedUserIdParam) {
        params.push(parseInt(viewedUserIdParam, 10));
        where.push(`s.user_id = $${params.length}`);
      } 
      else if (currentUser) {
        params.push(currentUser.id);
        where.push(`(s.user_id = $${params.length} OR s.status = 'available')`);
      }
      else {
        where.push(`s.status = 'available'`);
      }
    }

    if (where.length) {
      query += ' WHERE ' + where.join(' AND ');
    }
    query += ' ORDER BY s.shift_date, s.shift_code, s.user_id'; // ИЗМЕНЕНО: добавлена сортировка по user_id
    
    const result = await pool.query(query, params);
    const rows = result.rows.map(row => ({
        ...row,
        shift_date: format(new Date(row.shift_date), 'yyyy-MM-dd')
    }));

    return NextResponse.json(rows);
  } catch (err) {
    console.error('[GET /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  console.log('[POST /api/shifts] ========== REQUEST START ==========');
  
  try {
    const body = await request.json();
    console.log('[POST /api/shifts] Body:', body);
    
    const currentUser = await getUserFromRequest(request);
    console.log('[POST /api/shifts] Current user:', currentUser);

    if (!currentUser) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // УДАЛЯЕМ ПРОВЕРКУ isOwner - теперь любой авторизованный пользователь может создавать слоты
    // if (!currentUser.isOwner) {
    //   return NextResponse.json({ error: 'Forbidden: Only the owner can create shifts.' }, { status: 403 });
    // }

    const { date, startTime, endTime, assignToSelf } = body;
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
      
      // Проверяем, существует ли уже такой слот для этого пользователя
      const existing = await client.query(
        'SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2 AND user_id = $3',
        [dateStr, shiftCode, currentUser.id]
      );

      if (existing.rowCount && existing.rowCount > 0) {
        await client.query('COMMIT');
        console.log('[POST /api/shifts] User already has this shift:', existing.rows[0]);
        return NextResponse.json(existing.rows[0], { status: 200 });
      }
      
      // Создаем новый слот
      // assignToSelf всегда true для обычных пользователей
      const userId = assignToSelf !== false ? currentUser.id : null;
      const status = userId ? 'pending' : 'available';
      
      const insertQuery = `
        INSERT INTO shifts (shift_date, day_of_week, shift_code, status, user_id)
        VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, $3, $4)
        RETURNING *
      `;
      
      const result = await client.query(insertQuery, [dateStr, shiftCode, status, userId]);
      
      await client.query('COMMIT');
      
      console.log('[POST /api/shifts] New shift created:', result.rows[0]);
      return NextResponse.json(result.rows[0], { status: 201 });

    } catch (err: any) {
      await client.query('ROLLBACK');
      
      if (err.code === '23505') {
        console.warn('[POST /api/shifts] Duplicate shift constraint violation');
        return NextResponse.json({ error: 'Этот слот уже существует' }, { status: 409 });
      }
      
      console.error('[POST /api/shifts] Database Error:', err);
      throw err;
    } finally {
      client.release();
    }
  } catch (err) {
    console.error('[POST /api/shifts] General Error:', err);
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
    
    // Пользователи могут удалять только свои слоты
    console.log(`[API Shifts DELETE] User ${currentUser.id} clearing their shifts for ${dateStr}`);
    const deleteResult = await client.query(
      `DELETE FROM shifts WHERE shift_date = $1 AND user_id = $2`,
      [dateStr, currentUser.id]
    );
    
    console.log(`[API Shifts DELETE] Deleted ${deleteResult.rowCount} shifts.`);

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