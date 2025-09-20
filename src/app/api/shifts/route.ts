// src/app/api/shifts/route.ts (ИСПРАВЛЕННАЯ И КОНЕЧНАЯ ВЕРСИЯ)

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
            const { rows } = await pool.query<User>('SELECT id, username, full_name FROM users WHERE api_key = $1', [apiKey]);
            if (rows.length > 0) {
                // Добавляем isOwner на основе id
                return { ...rows[0], isOwner: rows[0].id === 1 };
            }
        }
    }

    try {
        const cookie = cookies().get('auth-session');
        if (!cookie) return null;
        const parsed = JSON.parse(cookie.value);
        if (parsed?.id) {
            // Добавляем isOwner на основе id
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

// GET-запросы не меняем, они выглядят корректно
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
    
    // Логика для владельца: он видит свои слоты и все доступные
    if (currentUser?.isOwner) {
       if (viewedUserIdParam && String(currentUser.id) !== viewedUserIdParam) {
           // Владелец смотрит чужой профиль - показываем только слоты этого юзера
           params.push(parseInt(viewedUserIdParam, 10));
           where.push(`s.user_id = $${params.length}`);
       } else {
           // Владелец смотрит свой профиль или общий вид
           params.push(currentUser.id);
           where.push(`(s.user_id = $${params.length} OR s.status = 'available')`);
       }
    } 
    // Логика для обычного пользователя: он видит только свои слоты и доступные
    else if (currentUser) {
        params.push(currentUser.id);
        where.push(`(s.user_id = $${params.length} OR s.status = 'available')`);
    }
    // Логика для неавторизованного запроса: видит только доступные слоты
    else {
        where.push(`s.status = 'available'`);
    }


    if (where.length) {
      query += ' WHERE ' + where.join(' AND ');
    }
    query += ' ORDER BY s.shift_date, s.shift_code';
    
    const result = await pool.query(query, params);
    // Даты могут приходить в формате ISODate, нормализуем их
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

// ===================================================================
// ГЛАВНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ
// ===================================================================
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

    // ИСПРАВЛЕНИЕ 1: Проверяем, что только владелец (id=1) может создавать слоты
    if (!currentUser.isOwner) {
      return NextResponse.json({ error: 'Forbidden: Only the owner can create shifts.' }, { status: 403 });
    }

    const { date, startTime, endTime, assignToSelf } = body;
    if (!date || !startTime || !endTime) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const dateStr = normalizeDateString(date);
    if (!dateStr) {
      return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
    }
    
    const shiftCode = `${startTime}-${endTime}`;
    const userId = assignToSelf ? currentUser.id : null;
    // Если слот создается для себя, он сразу занят. Если как свободный - доступен.
    const status = assignToSelf ? 'taken' : 'available'; 
    
    const client = await pool.connect();
    try {
      // ИСПРАВЛЕНИЕ 2: УДАЛЕНА РУЧНАЯ ПРОВЕРКА. Мы просто пытаемся вставить данные.
      // База данных сама не позволит создать дубликат для ОДНОГО И ТОГО ЖЕ пользователя.
      
      const insertQuery = `
        INSERT INTO shifts (shift_date, day_of_week, shift_code, status, user_id)
        VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, $3, $4)
        RETURNING *
      `;
      
      const result = await client.query(insertQuery, [dateStr, shiftCode, status, userId]);
      
      console.log('[POST /api/shifts] New shift created:', result.rows[0]);
      return NextResponse.json(result.rows[0], { status: 201 });

    } catch (err: any) {
      // ИСПРАВЛЕНИЕ 3: ЛОВИМ ОШИБКУ ОТ БАЗЫ ДАННЫХ
      // Если мы пытаемся создать дубликат для того же юзера, БД вернет ошибку 23505
      if (err.code === '23505' && err.constraint === 'shifts_user_date_code_unique') {
        console.warn('[POST /api/shifts] Attempted to create a duplicate shift for the same user.');
        return NextResponse.json({ error: 'Этот слот для данного пользователя уже существует' }, { status: 409 }); // 409 Conflict
      }
      
      // Если другая ошибка, то это проблема сервера
      console.error('[POST /api/shifts] Database Error:', err);
      throw err; // Передаем ошибку во внешний обработчик
    } finally {
      client.release();
    }
  } catch (err) {
    // Внешний обработчик ловит все остальные ошибки
    console.error('[POST /api/shifts] General Error:', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}


// DELETE-запросы не меняем, они выглядят корректно
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
    
    // Важная логика: Владелец может удалять все слоты за день, обычный юзер - только свои
    let deleteResult;
    if (currentUser.isOwner) {
        console.log(`[API Shifts DELETE] Owner ${currentUser.id} clearing ALL shifts for ${dateStr}`);
        deleteResult = await client.query(
            `DELETE FROM shifts WHERE shift_date = $1`, [dateStr]
        );
    } else {
        console.log(`[API Shifts DELETE] User ${currentUser.id} clearing THEIR shifts for ${dateStr}`);
        deleteResult = await client.query(
            `DELETE FROM shifts WHERE shift_date = $1 AND user_id = $2`, [dateStr, currentUser.id]
        );
    }
    
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