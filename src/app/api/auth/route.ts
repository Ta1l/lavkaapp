// src/app/api/auth/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';
import bcrypt from 'bcrypt'; // [ДОБАВЛЕНО]

const SALT_ROUNDS = 10; // Стандартное значение для "сложности" хеширования

export async function POST(request: NextRequest) {
    try {
        const { username, password, action } = await request.json();

        if (!username || !password || !action) {
            return NextResponse.json({ error: 'Не все поля заполнены' }, { status: 400 });
        }

        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser = existingUserResult.rowCount > 0 ? existingUserResult.rows[0] : null;

        let user: User | null = null;

        if (action === 'register') {
            if (existingUser) {
                return NextResponse.json({ error: 'Пользователь с таким именем уже существует' }, { status: 409 });
            }
            
            // [ИЗМЕНЕНО] Хешируем пароль перед сохранением
            const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
            
            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, hashedPassword] // Сохраняем хеш, а не пароль
            );
            user = newUserResult.rows[0];

        } else if (action === 'login') {
            if (!existingUser) {
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }

            // [ИЗМЕНЕНО] Сравниваем введенный пароль с хешем из базы
            const isPasswordCorrect = await bcrypt.compare(password, existingUser.password);

            if (!isPasswordCorrect) {
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            user = existingUser;
            
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
            // Возвращаем успешный ответ, но без лишних данных
            return NextResponse.json({ message: 'Успешно!' });
        }

        // Эта ветка не должна срабатывать, но это защита на всякий случай
        return NextResponse.json({ error: 'Произошла непредвиденная ошибка' }, { status: 500 });
    } catch (error) {
        console.error('[API Auth Error]', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}