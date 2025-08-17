// src/app/api/shifts/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { pool } from '@/lib/db';
import { format } from 'date-fns';
import { User } from '@/types/shifts';

// --- Вспомогательные функции (взяты из вашей версии, они отличные) ---

async function getUserFromSession(): Promise<User | null> {
  try {
    const cookie = cookies().get('auth-session');
    if (!cookie) return null;
    const parsed = JSON.parse(cookie.value);
    return parsed?.id ? parsed : null;
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

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const viewedUserIdParam = url.searchParams.get('userId');
    const startParam = url.searchParams.get('start');
    const endParam = url.searchParams.get('end');

    const startStr = startParam ? normalizeDateString(startParam) : null;
    const endStr = endParam ? normalizeDateString(endParam) : null;

    const currentUser = await getUserFromSession();
    
    // Динамическое построение запроса для гибкости
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
    
    // [ИСПРАВЛЕНА ЛОГИКА] Теперь она соответствует нашей архитектуре
    const isOwnerView = !viewedUserIdParam || (currentUser && currentUser.id.toString() === viewedUserIdParam);

    if (isOwnerView && currentUser) {
      // Я смотрю свое расписание: вижу свои слоты + все доступные
      params.push(currentUser.id);
      where.push(`(s.user_id = $${params.length} OR s.status = 'available')`);
    } else if (viewedUserIdParam) {
      // Я смотрю чужое расписание: вижу только занятые им слоты
      params.push(parseInt(viewedUserIdParam, 10));
      where.push(`s.user_id = $${params.length}`);
    } else {
      // Я не авторизован: вижу только доступные
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
  // Эта функция теперь отвечает ТОЛЬКО за создание нового ТИПА слота на день
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const { date, startTime, endTime } = await request.json();
    if (!date || !startTime || !endTime) return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });

    const dateStr = normalizeDateString(date);
    if (!dateStr) return NextResponse.json({ error: 'Invalid date' }, { status: 400 });

    const shiftCode = `${startTime}-${endTime}`;

    // Сначала ищем. Если нашли - просто возвращаем.
    const existing = await pool.query('SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2', [dateStr, shiftCode]);
    if (existing.rowCount > 0) {
        return NextResponse.json(existing.rows[0]);
    }

    // Если не нашли - создаем.
    const insert = await pool.query(
      `INSERT INTO shifts (shift_date, day_of_week, shift_code, status)
       VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, 'available')
       RETURNING *`,
      [dateStr, shiftCode]
    );
    return NextResponse.json(insert.rows[0], { status: 201 });
  } catch (err) {
    console.error('[POST /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  // Эта функция теперь отвечает ТОЛЬКО за ОСВОБОЖДЕНИЕ всех слотов за день
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

    const url = new URL(request.url);
    const dateParam = url.searchParams.get('date');
    if (!dateParam) return NextResponse.json({ error: 'Missing date param' }, { status: 400 });

    const dateStr = normalizeDateString(dateParam);
    if (!dateStr) return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
    
    // [ИСПРАВЛЕНО] Мы не удаляем, а ОСВОБОЖДАЕМ слоты
    const res = await pool.query(
      `UPDATE shifts SET user_id = NULL, status = 'available'
       WHERE shift_date = $1 AND user_id = $2`,
      [dateStr, currentUser.id]
    );
    return NextResponse.json({ ok: true, releasedCount: res.rowCount });
  } catch (err) {
    console.error('[DELETE /api/shifts] error', err);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}