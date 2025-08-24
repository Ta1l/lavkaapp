// src/app/api/auth/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';
import bcrypt from 'bcryptjs'; // [ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ] Заменено на bcryptjs
import crypto from 'crypto';

const SALT_ROUNDS = 10;
const BCRYPT_PREFIX_REGEX = /^\$2[aby]\$\d{2}\$/;

export async function POST(request: NextRequest) {
    try {
        const { username, password, action } = await request.json();
        if (!username || !password || !action) {
            return NextResponse.json({ error: 'Требуется username, password и action' }, { status: 400 });
        }

        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser: (User & { password: string, api_key: string }) | null = existingUserResult?.rows[0] || null;

        if (action === 'register') {
            if (existingUser) return NextResponse.json({ error: 'Пользователь уже существует' }, { status: 409 });
            const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, hashedPassword]
            );
            const user = newUserResult.rows[0];
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), { httpOnly: true, secure: process.env.NODE_ENV === 'production', maxAge: 60 * 60 * 24 * 7, path: '/' });
            return NextResponse.json({ message: 'Регистрация успешна!' });
        }

        if (action === 'login') {
            if (!existingUser) return NextResponse.json({ error: 'Неверные учетные данные' }, { status: 401 });

            let isPasswordCorrect = false;
            const passwordInDb = existingUser.password;

            if (BCRYPT_PREFIX_REGEX.test(passwordInDb)) {
                isPasswordCorrect = await bcrypt.compare(password, passwordInDb);
            } else {
                if (password === passwordInDb) {
                    isPasswordCorrect = true;
                    const newHashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
                    await pool.query('UPDATE users SET password = $1 WHERE id = $2', [newHashedPassword, existingUser.id]);
                }
            }

            if (!isPasswordCorrect) return NextResponse.json({ error: 'Неверные учетные данные' }, { status: 401 });

            const user = existingUser;
            let apiKey = user.api_key;
            if (!apiKey) {
                apiKey = crypto.randomBytes(16).toString('hex');
                await pool.query("UPDATE users SET api_key = $1 WHERE id = $2", [apiKey, user.id]);
            }
            // Устанавливаем cookie и для логина тоже
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), { httpOnly: true, secure: process.env.NODE_ENV === 'production', maxAge: 60 * 60 * 24 * 7, path: '/' });
            
            return NextResponse.json({ message: 'Вход успешен!', apiKey });
        }
        
        return NextResponse.json({ error: 'Неверное действие' }, { status: 400 });
    } catch (error) {
        console.error('[API Auth Error]', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}