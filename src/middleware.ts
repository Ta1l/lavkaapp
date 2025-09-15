// src/middleware.ts

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Список путей, которые не требуют аутентификации через cookie
const publicPaths = [
    '/',             // Главная страница (форма входа)
    '/api/auth',     // API для входа, регистрации, выхода
    '/api/shifts',   // API для получения списка смен
    '/api/slots',    // API для управления слотами
];

export function middleware(request: NextRequest) {
    const path = request.nextUrl.pathname;

    // Проверяем, является ли текущий путь публичным
    const isPublicPath = publicPaths.some(publicPath =>
        path.startsWith(publicPath)
    );
    
    // Если путь публичный, пропускаем
    if (isPublicPath) {
        return NextResponse.next();
    }
    
    // Для API запросов с Bearer токеном тоже пропускаем
    if (path.startsWith('/api/') && request.headers.get('authorization')?.startsWith('Bearer ')) {
        return NextResponse.next();
    }
    
    // Проверка cookie для защищенных путей
    const sessionCookie = request.cookies.get('auth-session');

    if (!sessionCookie) {
        // Для API возвращаем 401, для страниц - редирект
        if (path.startsWith('/api/')) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }
        return NextResponse.redirect(new URL('/', request.url));
    }
    
    return NextResponse.next();
}

export const config = {
    matcher: [
        '/((?!_next/static|_next/image|favicon.ico).*)',
    ],
};