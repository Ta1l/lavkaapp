// src/app/api/auth/get-token/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import bcrypt from 'bcryptjs';
import { v4 as uuidv4 } from 'uuid';

const BCRYPT_PREFIX_REGEX = /^\$2[aby]\$\d{2}\$/;

export async function POST(request: NextRequest) {
  try {
    const { username, password } = await request.json();
    const telegramId = request.headers.get('x-telegram-id');
    
    console.log('[GET-TOKEN] Request:', { username, hasTelegramId: !!telegramId });

    if (!username || !password) {
      return NextResponse.json({ error: 'Username and password required' }, { status: 400 });
    }

    // Проверяем пользователя
    const userResult = await pool.query(
      'SELECT id, username, password, api_key, telegram_id FROM users WHERE username = $1',
      [username]
    );

    if (userResult.rowCount === 0) {
      return NextResponse.json({ error: 'User not found' }, { status: 404 });
    }

    const user = userResult.rows[0];

    // Проверяем пароль
    let isPasswordCorrect = false;
    const passwordInDb = user.password;
    
    if (typeof passwordInDb === 'string' && passwordInDb.length > 0) {
      if (BCRYPT_PREFIX_REGEX.test(passwordInDb)) {
        isPasswordCorrect = await bcrypt.compare(password, passwordInDb);
      } else {
        isPasswordCorrect = password === passwordInDb;
      }
    }

    if (!isPasswordCorrect) {
      return NextResponse.json({ error: 'Invalid password' }, { status: 401 });
    }

    // Генерируем API ключ если его нет
    let apiKey = user.api_key;
    if (!apiKey) {
      apiKey = uuidv4();
      await pool.query(
        'UPDATE users SET api_key = $1 WHERE id = $2',
        [apiKey, user.id]
      );
    }

    // Связываем Telegram ID если он передан и еще не связан
    if (telegramId && !user.telegram_id) {
      console.log('[GET-TOKEN] Linking Telegram ID:', telegramId, 'to user:', user.id);
      await pool.query(
        'UPDATE users SET telegram_id = $1 WHERE id = $2',
        [telegramId, user.id]
      );
    }

    return NextResponse.json({ apiKey }, { status: 200 });
  } catch (err) {
    console.error('[GET-TOKEN Error]', err);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}