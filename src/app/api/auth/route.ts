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
        if (!username || !password) return NextResponse.json({ error: 'Не все поля заполнены' }, { status: 400 });

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

            // Проверяем, является ли пароль в базе хешем
            if (BCRYPT_PREFIX_REGEX.test(passwordInDb)) {
                // --- СЦЕНАРИЙ 1: ПАРОЛЬ УЖЕ ХЕШИРОВАН ---
                console.log(`[AUTH] Пользователь ${username}: пароль в БД - хеш. Сравниваем через bcrypt.`);
                isPasswordCorrect = await bcrypt.compare(password, passwordInDb);
            } else {
                // --- СЦЕНАРИЙ 2: ПАРОЛЬ В ОТКРЫТОМ ВИДЕ (СТАРЫЙ ПОЛЬЗОВАТЕЛЬ) ---
                console.log(`[AUTH] Пользователь ${username}: пароль в БД - открытый текст. Сравниваем напрямую.`);
                if (password === passwordInDb) {
                    isPasswordCorrect = true;
                    console.log(`[AUTH] Пароли совпали. Проводим "ленивую" миграцию для пользователя ${username}...`);
                    // Немедленно хешируем и обновляем пароль в базе
                    const newHashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
                    await pool.query('UPDATE users SET password = $1 WHERE id = $2', [newHashedPassword, existingUser.id]);
                    console.log(`[AUTH] Пароль для пользователя ${username} успешно обновлен до хеша.`);
                }
            }

            if (!isPasswordCorrect) {
                return NextResponse.json({ error: 'Неверные учетные данные' }, { status: 401 });
            }

            const user = existingUser;

            // Устанавливаем cookie сессии
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), { httpOnly: true, secure: process.env.NODE_ENV === 'production', maxAge: 60 * 60 * 24 * 7, path: '/' });
            
            // Генерируем/возвращаем apiKey для бота
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