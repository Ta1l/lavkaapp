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

    let parsed: any;
    try {
      parsed = JSON.parse(cookie.value);
    } catch {
      return null;
    }

    if (!parsed?.id) return null;

    const res = await pool.query(
      'SELECT id, username, full_name FROM users WHERE id = $1',
      [parsed.id]
    );

    if (res.rowCount === 0) return null;
    return res.rows[0];
  } catch (err) {
    console.error('[getUserFromSession] error', err);
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

/**
 * GET - Получение смен для отображения в календаре.
 * Умеет фильтровать по диапазону дат и по пользователю.
 * Логика взята из /api/slots и улучшена.
 */
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
      // Важно: `<` а не `<=`, чтобы не включать следующий день
      where.push(`s.shift_date < $${params.length}`);
    }

    if (viewedUserIdParam) {
      const vid = parseInt(viewedUserIdParam, 10);
      if (Number.isNaN(vid)) {
        return NextResponse.json({ error: 'Invalid userId' }, { status: 400 });
      }
      params.push(vid);
      where.push(`s.user_id = $${params.length}`);
    } else if (currentUser) {
      // Если смотрим свой календарь, показываем свои смены + доступные для взятия
      params.push(currentUser.id);
      where.push(`(s.user_id = $${params.length} OR s.status = 'available')`);
    } else {
      // Если неавторизованный пользователь, показываем только доступные
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

/**
 * POST - Создание новой смены или взятие существующей свободной.
 * Логика взята из /api/slots.
 */
export async function POST(request: NextRequest) {
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await request.json();
    const { date, startTime, endTime, status } = body || {};

    if (!date || !startTime || !endTime) {
      return NextResponse.json({ error: 'Missing required fields: date, startTime, endTime' }, { status: 400 });
    }

    const dateStr = normalizeDateString(date);
    if (!dateStr) {
      return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
    }

    const shiftCode = `${startTime}-${endTime}`;
    const shiftStatus = status && typeof status === 'string' ? status : 'pending';

    // Проверяем, существует ли уже такой слот
    const existing = await pool.query(
      `SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2`,
      [dateStr, shiftCode]
    );

    if (existing.rowCount > 0) {
      const slot = existing.rows[0];
      // Если это уже наш слот, ничего не делаем
      if (slot.user_id === currentUser.id) {
        return NextResponse.json(slot);
      }
      // Если слот свободен, занимаем его
      if (!slot.user_id || slot.status === 'available') {
        const updated = await pool.query(
          `UPDATE shifts
           SET user_id = $1, status = 'pending'
           WHERE id = $2
           RETURNING id, user_id, shift_date, shift_code, status`,
          [currentUser.id, slot.id]
        );
        return NextResponse.json(updated.rows[0]);
      }
      // Если слот занят кем-то другим
      return NextResponse.json({ error: 'Slot already taken' }, { status: 409 });
    }

    // Если слота не существует, создаем новый
    const insert = await pool.query(
      `INSERT INTO shifts (user_id, shift_date, day_of_week, shift_code, status)
       VALUES ($1, $2::date, extract(dow from $2::date)::int + 1, $3, $4)
       RETURNING id, user_id, shift_date, shift_code, status`,
      [currentUser.id, dateStr, shiftCode, shiftStatus]
    );

    return NextResponse.json(insert.rows[0], { status: 201 });
  } catch (err) {
    console.error('[POST /api/shifts] error', err);
    // Обработка уникального констрейнта
    if (err.code === '23505') {
       return NextResponse.json({ error: 'A shift with this date and time already exists.' }, { status: 409 });
    }
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

/**
 * PUT - Обновление существующей смены.
 * Логика взята из /api/slots.
 */
export async function PUT(request: NextRequest) {
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    const body = await request.json();
    const { id, date, startTime, endTime, status } = body || {};

    if (!id) {
      return NextResponse.json({ error: 'Missing shift id' }, { status: 400 });
    }

    const existing = await pool.query('SELECT * FROM shifts WHERE id = $1', [id]);

    if (existing.rowCount === 0) {
      return NextResponse.json({ error: 'Shift not found' }, { status: 404 });
    }

    const slot = existing.rows[0];
    if (slot.user_id !== currentUser.id) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const updates: string[] = [];
    const params: any[] = [];
    let idx = 1;

    if (date) {
      const dateStr = normalizeDateString(date);
      if (!dateStr) {
        return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
      }
      updates.push(`shift_date = $${idx++}`);
      params.push(dateStr);
      updates.push(`day_of_week = extract(dow from $${idx - 1}::date)::int + 1`);
    }
    if (startTime && endTime) {
      const shiftCode = `${startTime}-${endTime}`;
      updates.push(`shift_code = $${idx++}`);
      params.push(shiftCode);
    }
    if (status) {
      updates.push(`status = $${idx++}`);
      params.push(status);
    }

    if (!updates.length) {
      return NextResponse.json({ error: 'Nothing to update' }, { status: 400 });
    }
    params.push(id);
    
    const res = await pool.query(
      `UPDATE shifts SET ${updates.join(', ')} WHERE id = $${params.length}
       RETURNING id, user_id, shift_date, shift_code, status`,
      params
    );
    return NextResponse.json(res.rows[0]);
  } catch (err) {
    console.error('[PUT /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}


/**
 * DELETE — Удаление смены.
 * Поддерживает удаление по id (или alias slotId) или по дате.
 */
export async function DELETE(request: NextRequest) {
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const url = new URL(request.url);
    // Принимаем 'id' или 'slotId' для совместимости
    const idParam = url.searchParams.get('id') || url.searchParams.get('slotId');
    const dateParam = url.searchParams.get('date');

    if (!idParam && !dateParam) {
      return NextResponse.json({ error: 'Missing id/slotId or date param' }, { status: 400 });
    }

    if (idParam) {
      const id = parseInt(idParam, 10);
      if (Number.isNaN(id)) {
        return NextResponse.json({ error: 'Invalid id' }, { status: 400 });
      }
      
      const existing = await pool.query('SELECT * FROM shifts WHERE id = $1', [id]);
      if (existing.rowCount === 0) {
        return NextResponse.json({ error: 'Shift not found' }, { status: 404 });
      }
      
      const slot = existing.rows[0];
      // Удалять может только владелец смены
      if (slot.user_id !== currentUser.id) {
        return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
      }
      
      await pool.query('DELETE FROM shifts WHERE id = $1', [id]);
      return NextResponse.json({ ok: true });
    }

    if (dateParam) {
      const dateStr = normalizeDateString(dateParam);
      if (!dateStr) {
        return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
      }
      // Удаляем все смены текущего юзера за указанную дату
      const res = await pool.query(
        'DELETE FROM shifts WHERE shift_date = $1 AND user_id = $2 RETURNING id',
        [dateStr, currentUser.id]
      );
      return NextResponse.json({
        ok: true,
        deletedCount: res.rowCount
      });
    }

  } catch (err) {
    console.error('[DELETE /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}