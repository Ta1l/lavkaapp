// src/app/api/auth/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { pool } from '@/lib/db';
import bcrypt from 'bcrypt';

const SALT_ROUNDS = 10;

type DBUser = {
  id: number;
  username: string;
  full_name: string | null;
  password: string; // присутствует только из БД
};

function setSessionCookieAndRedirect(req: NextRequest, user: { id: number; username: string }) {
  const res = NextResponse.redirect(new URL('/schedule/0', req.url));
  res.cookies.set('auth-session', JSON.stringify(user), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7, // 7 дней
    path: '/',
  });
  return res;
}

function setSessionCookieJSON(user: { id: number; username: string }) {
  const res = NextResponse.json({ ok: true, redirectTo: '/schedule/0' });
  res.cookies.set('auth-session', JSON.stringify(user), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7,
    path: '/',
  });
  return res;
}

export async function POST(request: NextRequest) {
  try {
    // Поддержка как JSON, так и обычной HTML-формы
    const contentType = request.headers.get('content-type') || '';
    let username = '';
    let password = '';
    let action = '';

    if (contentType.includes('application/json')) {
      const body = await request.json();
      username = (body?.username ?? '').trim();
      password = body?.password ?? '';
      action = body?.action ?? '';
    } else {
      const form = await request.formData();
      username = String(form.get('username') ?? '').trim();
      password = String(form.get('password') ?? '');
      action = String(form.get('action') ?? '');
    }

    if (!username || !password || (action !== 'login' && action !== 'register')) {
      const msg = 'Не все поля заполнены или неверное действие';
      return NextResponse.json({ error: msg }, { status: 400 });
    }

    const existing = await pool.query('SELECT id, username, full_name, password FROM users WHERE username = $1', [username]);
    const existingUser: DBUser | null = existing.rowCount ? (existing.rows[0] as DBUser) : null;

    if (action === 'register') {
      if (existingUser) {
        return NextResponse.json({ error: 'Пользователь с таким именем уже существует' }, { status: 409 });
      }

      const hashed = await bcrypt.hash(password, SALT_ROUNDS);
      const ins = await pool.query(
        'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
        [username, hashed]
      );
      const user = ins.rows[0] as { id: number; username: string; full_name: string | null };

      // Если это был JSON-запрос (например, мобильный клиент) — вернём JSON + Set-Cookie,
      // если форма — отдадим редирект.
      if (contentType.includes('application/json')) {
        return setSessionCookieJSON({ id: user.id, username: user.username });
      }
      return setSessionCookieAndRedirect(request, { id: user.id, username: user.username });
    }

    // action === 'login'
    if (!existingUser) {
      return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
    }

    const ok = await bcrypt.compare(password, existingUser.password);
    if (!ok) {
      return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
    }

    if (contentType.includes('application/json')) {
      return setSessionCookieJSON({ id: existingUser.id, username: existingUser.username });
    }
    return setSessionCookieAndRedirect(request, { id: existingUser.id, username: existingUser.username });
  } catch (err) {
    console.error('[API /api/auth] error:', err);
    return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
  }
}
