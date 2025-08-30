// src/app/api/auth/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User } from '@/types/shifts';
import bcrypt from 'bcryptjs';

const SALT_ROUNDS = 10;
const BCRYPT_PREFIX_REGEX = /^\$2[aby]\$\d{2}\$/;

// [ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ] Получаем наш публичный URL из переменных окружения
const SITE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://slotworker.ru';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const username = formData.get('username') as string;
        const password = formData.get('password') as string;
        const action = formData.get('action') as 'login' | 'register';

        if (!username || !password || !action) {
            return NextResponse.redirect(new URL('/?error=validation_failed', SITE_URL));
        }

        const existingUserResult = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
        const existingUser: (User & { password: string }) | null = existingUserResult?.rows[0] || null;

        if (action === 'register') {
            if (existingUser) {
                return NextResponse.redirect(new URL('/?error=user_exists', SITE_URL));
            }
            const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
            const newUserResult = await pool.query(
                'INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username, full_name',
                [username, hashedPassword]
            );
            const user = newUserResult.rows[0];
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), { httpOnly: true, secure: true, maxAge: 60 * 60 * 24 * 7, path: '/' });
            return NextResponse.redirect(new URL('/schedule/0', SITE_URL));
        }

        if (action === 'login') {
            if (!existingUser) {
                return NextResponse.redirect(new URL('/?error=invalid_credentials', SITE_URL));
            }
            
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

            if (!isPasswordCorrect) {
                return NextResponse.redirect(new URL('/?error=invalid_credentials', SITE_URL));
            }

            const user = existingUser;
            const sessionData = { id: user.id, username: user.username };
            cookies().set('auth-session', JSON.stringify(sessionData), { httpOnly: true, secure: true, maxAge: 60 * 60 * 24 * 7, path: '/' });
            return NextResponse.redirect(new URL('/schedule/0', SITE_URL));
        }
        
        return NextResponse.redirect(new URL('/?error=invalid_action', SITE_URL));
    } catch (error) {
        console.error('[API Auth Error]', error);
        return NextResponse.redirect(new URL('/?error=server_error', SITE_URL));
    }
}