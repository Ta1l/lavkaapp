// src/app/api/auth/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';
import bcrypt from 'bcrypt';

const SALT_ROUNDS = 10;

export async function POST(request: NextRequest) {
    try {
        // [ИЗМЕНЕНО] Читаем данные из FormData, а не из JSON
        const formData = await request.formData();
        const username = formData.get('username') as string;
        const password = formData.get('password') as string;
        const action = formData.get('action') as 'login' | 'register';

        if (!username || !password || !action) {
            // В случае ошибки, возвращаемся на главную с сообщением
            // (продвинутая версия могла бы передавать ошибку в URL)
            return NextResponse.redirect(new URL('/?error=validation_failed', request.url));
        }

        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser: (User & { password: string }) | null = existingUserResult?.rows[0] || null;

        let user: User | null = null;

        if (action === 'register') {
            if (existingUser) {
                return NextResponse.redirect(new URL('/?error=user_exists', request.url));
            }
            const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, hashedPassword]
            );
            user = newUserResult.rows[0];
        } else if (action === 'login') {
            if (!existingUser) {
                return NextResponse.redirect(new URL('/?error=invalid_credentials', request.url));
            }
            const isPasswordCorrect = await bcrypt.compare(password, existingUser.password);
            if (!isPasswordCorrect) {
                return NextResponse.redirect(new URL('/?error=invalid_credentials', request.url));
            }
            user = existingUser;
        } else {
             return NextResponse.redirect(new URL('/?error=invalid_action', request.url));
        }

        if (user) {
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), {
                httpOnly: true, secure: process.env.NODE_ENV === 'production', maxAge: 60 * 60 * 24 * 7, path: '/',
            });
            // [ИЗМЕНЕНО] Сервер сам выполняет редирект
            return NextResponse.redirect(new URL('/schedule/0', request.url));
        }

        // Если что-то пошло не так, возвращаемся на главную
        return NextResponse.redirect(new URL('/?error=unknown', request.url));

    } catch (error) {
        console.error('[API Auth Error]', error);
        return NextResponse.redirect(new URL('/?error=server_error', request.url));
    }
}