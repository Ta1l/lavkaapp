// src/app/api/auth/route.ts

import { NextRequest, NextResponse } from 'next/server';
// [ИЗМЕНЕНО] - Импортируем 'pool' напрямую, а не несуществующую функцию 'getDbPool'
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';

export async function POST(request: NextRequest) {
    try {
        const { username, password, action } = await request.json();

        if (!username || !password || !action) {
            return NextResponse.json({ error: 'Не все поля заполнены' }, { status: 400 });
        }

        // [ИЗМЕНЕНО] - 'pool' уже является готовым объектом, его не нужно вызывать как функцию.
        // Строка 'const pool = getDbPool();' была удалена.
        const existingUser = await pool.query('SELECT * FROM users WHERE username = $1', [username]);

        let user: User | null = null;

        if (action === 'register') {
            if (existingUser.rowCount > 0) {
                return NextResponse.json({ error: 'Пользователь с таким именем уже существует' }, { status: 409 });
            }
            // Здесь должна быть логика хеширования пароля, но пока оставляем как есть
            const newUser = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, password]
            );
            user = newUser.rows[0];
        } else if (action === 'login') {
            if (existingUser.rowCount === 0) {
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            // Здесь должна быть логика сравнения хеша пароля, но пока оставляем как есть
            if (existingUser.rows[0].password !== password) {
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            user = existingUser.rows[0];
        } else {
            return NextResponse.json({ error: 'Неверное действие' }, { status: 400 });
        }

        if (user) {
            // Создаем сессию в cookie
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                maxAge: 60 * 60 * 24 * 7, // 1 неделя
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