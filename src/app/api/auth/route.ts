// src/app/api/auth/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';

export async function POST(request: NextRequest) {
    console.log('\n--- [SERVER LOG] /api/auth: Получен POST-запрос ---');
    try {
        const { username, password, action } = await request.json();
        console.log(`[SERVER LOG] Данные из запроса: action=${action}, username=${username}`);

        if (!username || !password || !action) {
            console.error('[SERVER LOG] Ошибка: Не все поля заполнены. Отправка 400.');
            return NextResponse.json({ error: 'Не все поля заполнены' }, { status: 400 });
        }

        type UserWithPassword = User & { password: string };

        console.log(`[SERVER LOG] Поиск пользователя "${username}" в базе данных...`);
        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser: UserWithPassword | null = existingUserResult?.rows[0] || null;

        if (existingUser) {
            console.log(`[SERVER LOG] Пользователь "${username}" найден. ID: ${existingUser.id}`);
        } else {
            console.log(`[SERVER LOG] Пользователь "${username}" не найден.`);
        }

        let user: User | null = null;

        if (action === 'register') {
            console.log('[SERVER LOG] Выполняется действие: РЕГИСТРАЦИЯ');
            if (existingUser) {
                console.log(`[SERVER LOG] Конфликт: Пользователь "${username}" уже существует. Отправка 409.`);
                return NextResponse.json({ error: 'Пользователь с таким именем уже существует' }, { status: 409 });
            }

            console.log(`[SERVER LOG] Создание нового пользователя "${username}"...`);
            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, password]
            );
            user = newUserResult.rows[0];
            
            // [ИСПРАВЛЕНО] Добавлена проверка, чтобы TypeScript был уверен, что user не null
            if (user) {
                console.log(`[SERVER LOG] Пользователь "${username}" успешно создан. ID: ${user.id}`);
            } else {
                // Этот блок практически недостижим, но он нужен для безопасности
                throw new Error('RETURNING не вернул данные после INSERT');
            }

        } else if (action === 'login') {
            console.log('[SERVER LOG] Выполняется действие: ВХОД');
            if (!existingUser) {
                console.log(`[SERVER LOG] Ошибка входа: Пользователь "${username}" не найден. Отправка 401.`);
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            
            console.log('[SERVER LOG] Сравнение паролей...');
            if (existingUser.password !== password) {
                console.log('[SERVER LOG] Ошибка входа: Пароли НЕ совпадают. Отправка 401.');
                return NextResponse.json({ error: 'Неверное имя пользователя или пароль' }, { status: 401 });
            }
            
            console.log('[SERVER LOG] Пароли совпали.');
            user = existingUser;

        } else {
            console.log(`[SERVER LOG] Неверное действие: "${action}". Отправка 400.`);
            return NextResponse.json({ error: 'Неверное действие' }, { status: 400 });
        }

        if (user) {
            console.log(`[SERVER LOG] Успех. Установка cookie для пользователя ID: ${user.id}`);
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                maxAge: 60 * 60 * 24 * 7,
                path: '/',
            });
            console.log('[SERVER LOG] Отправка успешного ответа (200 OK).');
            return NextResponse.json({ message: 'Успешно!' });
        }

        console.error('[SERVER LOG] Критическая ошибка: пользователь null после всех проверок. Отправка 500.');
        return NextResponse.json({ error: 'Произошла непредвиденная ошибка' }, { status: 500 });

    } catch (error) {
        console.error('[SERVER LOG] КРИТИЧЕСКАЯ ОШИБКА в блоке try-catch:', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}