// src/middleware.ts

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Список путей, которые не требуют аутентификации
const publicPaths = [
    '/',             // Главная страница
    '/api/auth',     // [ИЗМЕНЕНО] Теперь это префикс, а не точный путь
    '/favicon.ico',
];

export function middleware(request: NextRequest) {
    const path = request.nextUrl.pathname;

    // [ИСПРАВЛЕНО] Логика проверки теперь использует startsWith для /api/auth
    // Это разрешит доступ и к /api/auth, и к /api/auth/get-token, и к /api/auth/logout
    const isPublicPath = publicPaths.some(publicPath =>
        path === publicPath || (publicPath === '/api/auth' && path.startsWith(publicPath))
    );

    const sessionCookie = request.cookies.get('auth-session');

    if (isPublicPath) {
        // Если путь публичный, ничего не делаем
        return NextResponse.next();
    }

    if (!sessionCookie) {
        // Если путь защищенный и нет сессии, редирект на главную
        return NextResponse.redirect(new URL('/', request.url));
    }
    
    // Если сессия есть, пропускаем
    return NextResponse.next();
}

// Конфигурация matcher остается без изменений
export const config = {
    matcher: [
        '/((?!_next/static|_next/image|favicon.ico).*)',
    ],
};