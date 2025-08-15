// src/app/api/auth/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';

export async function POST(request: NextRequest) {
    try {
        const { username, password, action } = await request.json();

        if (!username || !password || !action) {
            return NextResponse.json({ error: 'Не все поля заполнены' }, { status: 400 });
        }

        // Явно говорим, что в этом объекте есть пароль
        type UserWithPassword = User & { password: string };

        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser: UserWithPassword | null = existingUserResult?.rows[0] || null;

        let user: User | null = null;

        if (action === 'register') {
            if (existingUser) {
                return NextResponse.json({ error: 'Пользователь с таким именем уже существует' }, { status: 409 });
            }

            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, password]
            );
            user = newUserResult.rows[0];

        } else if (action === 'login') {
            if (!existingUser || existingUser.password !== password) {
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            user = existingUser;

        } else {
            return NextResponse.json({ error: 'Неверное действие' }, { status: 400 });
        }

        if (user) {
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                maxAge: 60 * 60 * 24 * 7,
                path: '/',
            });
            return NextResponse.json({ message: 'Успешно!' });
        }

        return NextResponse.json({ error: 'Произошла непредвиденная ошибка' }, { status: 500 });

    } catch (error) {
        console.error('[API Auth Error]', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}
