// src/app/api/auth/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';
import bcrypt from 'bcrypt';
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

        if (action === 'register') { /* ... код регистрации ... */ }

        if (action === 'login') {
            if (!existingUser) return NextResponse.json({ error: 'Неверные учетные данные' }, { status: 401 });

            let isPasswordCorrect = false;
            const passwordInDb = existingUser.password;

            try {
                if (BCRYPT_PREFIX_REGEX.test(passwordInDb)) {
                    isPasswordCorrect = await bcrypt.compare(password, passwordInDb);
                }
            } catch (e) { isPasswordCorrect = false; }

            if (!isPasswordCorrect && password === passwordInDb) {
                isPasswordCorrect = true;
                const newHashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
                await pool.query('UPDATE users SET password = $1 WHERE id = $2', [newHashedPassword, existingUser.id]);
            }

            if (!isPasswordCorrect) return NextResponse.json({ error: 'Неверные учетные данные' }, { status: 401 });

            const user = existingUser;
            let apiKey = user.api_key;
            if (!apiKey) {
                apiKey = crypto.randomBytes(16).toString('hex');
                await pool.query("UPDATE users SET api_key = $1 WHERE id = $2", [apiKey, user.id]);
            }
            return NextResponse.json({ message: 'Вход успешен!', apiKey });
        }
        
        return NextResponse.json({ error: 'Неверное действие' }, { status: 400 });
    } catch (error) {
        console.error('[API Auth Error]', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}