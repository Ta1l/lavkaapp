// src/app/api/auth/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';

const SALT_ROUNDS = 10;
const BCRYPT_PREFIX_REGEX = /^\$2[aby]\$\d{2}\$/;

export async function POST(request: NextRequest) {
    try {
        // [ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ] Читаем данные из FormData, а не из JSON
        const formData = await request.formData();
        const username = formData.get('username') as string;
        const password = formData.get('password') as string;
        const action = formData.get('action') as 'login' | 'register';

        if (!username || !password || !action) {
            return NextResponse.redirect(new URL('/?error=validation_failed', request.url));
        }

        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser: (User & { password: string, api_key: string }) | null = existingUserResult?.rows[0] || null;

        if (action === 'register') {
            if (existingUser) return NextResponse.redirect(new URL('/?error=user_exists', request.url));
            const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, hashedPassword]
            );
            const user = newUserResult.rows[0];
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), { httpOnly: true, secure: process.env.NODE_ENV === 'production', maxAge: 60 * 60 * 24 * 7, path: '/' });
            return NextResponse.redirect(new URL('/schedule/0', request.url));
        }

        if (action === 'login') {
            if (!existingUser) return NextResponse.redirect(new URL('/?error=invalid_credentials', request.url));
            
            let isPasswordCorrect = false;
            const passwordInDb = existingUser.password;

            if (typeof passwordInDb === 'string' && passwordInDb.length > 0) {
                if (BCRYPT_PREFIX_REGEX.test(passwordInDb)) {
                    isPasswordCorrect = await bcrypt.compare(password, passwordInDb);
                } else {
                    if (password === passwordInDb) {
                        isPasswordCorrect = true;
                        const newHashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
                        await pool.query('UPDATE users SET password = $1 WHERE id = $2', [newHashedPassword, existingUser.id]);
                    }
                }
            }

            if (!isPasswordCorrect) return NextResponse.redirect(new URL('/?error=invalid_credentials', request.url));

            const user = existingUser;
            let apiKey = user.api_key;
            if (!apiKey) {
                apiKey = crypto.randomBytes(16).toString('hex');
                await pool.query("UPDATE users SET api_key = $1 WHERE id = $2", [apiKey, user.id]);
            }
            
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), { httpOnly: true, secure: process.env.NODE_ENV === 'production', maxAge: 60 * 60 * 24 * 7, path: '/' });
            
            // Сервер сам выполняет редирект
            return NextResponse.redirect(new URL('/schedule/0', request.url));
        }
        
        return NextResponse.redirect(new URL('/?error=invalid_action', request.url));
    } catch (error) {
        console.error('[API Auth Error]', error);
        return NextResponse.redirect(new URL('/?error=server_error', request.url));
    }
}