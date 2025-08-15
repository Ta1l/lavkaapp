// src/app/api/auth/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';

export async function POST(request: NextRequest) {
    console.log('\n--- [СЕРВЕР] /api/auth: ПОЛУЧЕН ЗАПРОС ---');
    try {
        const { username, password, action } = await request.json();
        console.log(`[СЕРВЕР] ДАННЫЕ: action=${action}, username=${username}`);

        if (!username || !password || !action) {
            console.error('[СЕРВЕР] ОШИБКА: Не все поля. Ответ 400.');
            return NextResponse.json({ error: 'Не все поля заполнены' }, { status: 400 });
        }

        type UserWithPassword = User & { password: string };

        console.log(`[СЕРВЕР] ПОИСК пользователя "${username}" в БД...`);
        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser: UserWithPassword | null = existingUserResult?.rows[0] || null;

        if (existingUser) {
            console.log(`[СЕРВЕР] НАЙДЕН пользователь "${username}" с ID ${existingUser.id}. Пароль в БД: "${existingUser.password}"`);
        } else {
            console.log(`[СЕРВЕР] Пользователь "${username}" НЕ НАЙДЕН.`);
        }

        let user: User | null = null;

        if (action === 'register') {
            console.log('[СЕРВЕР] ДЕЙСТВИЕ: РЕГИСТРАЦИЯ');
            if (existingUser) {
                console.log(`[СЕРВЕР] КОНФЛИКТ. Пользователь уже есть. Ответ 409.`);
                return NextResponse.json({ error: 'Пользователь с таким именем уже существует' }, { status: 409 });
            }

            console.log(`[СЕРВЕР] СОЗДАНИЕ нового пользователя с паролем "${password}"...`);
            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, password]
            );
            user = newUserResult.rows[0];
            if(user) {
                console.log(`[СЕРВЕР] УСПЕХ. Пользователь создан с ID ${user.id}.`);
            } else {
                 throw new Error('Не удалось создать пользователя');
            }

        } else if (action === 'login') {
            console.log('[СЕРВЕР] ДЕЙСТВИЕ: ВХОД');
            if (!existingUser) {
                console.log(`[СЕРВЕР] ОТКАЗ. Пользователь не найден. Ответ 401.`);
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            
            console.log(`[СЕРВЕР] СРАВНЕНИЕ паролей: (введенный)"${password}" === (из БД)"${existingUser.password}"`);
            if (existingUser.password !== password) {
                console.log(`[СЕРВЕР] ОТКАЗ. Пароли не совпадают. Ответ 401.`);
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            
            console.log('[СЕРВЕР] УСПЕХ. Пароли совпали.');
            user = existingUser;

        } else {
            return NextResponse.json({ error: 'Неверное действие' }, { status: 400 });
        }

        if (user) {
            console.log(`[СЕРВЕР] УСТАНОВКА COOKIE для пользователя ID ${user.id}.`);
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                maxAge: 60 * 60 * 24 * 7,
                path: '/',
            });
            console.log('[СЕРВЕР] ОТВЕТ 200 OK.');
            return NextResponse.json({ message: 'Успешно!' });
        }
        
        console.error('[СЕРВЕР] КРИТИЧЕСКАЯ ОШИБКА: user=null. Ответ 500.');
        return NextResponse.json({ error: 'Произошла непредвиденная ошибка' }, { status: 500 });

    } catch (error) {
        console.error('[СЕРВЕР] КРИТИЧЕСКАЯ ОШИБКА В TRY-CATCH:', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}