// src/middleware.ts

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Список путей, которые не требуют аутентификации через cookie
const publicPaths = [
    '/',             // Главная страница (форма входа)
    '/api/auth',     // API для входа, регистрации, выхода
    
    // --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Добавляем API-пути для бота ---
    '/api/shifts',   // API для получения списка смен
    '/api/slots',    // API для управления слотами (добавление/удаление)
];

export function middleware(request: NextRequest) {
    const path = request.nextUrl.pathname;

    // Проверяем, является ли текущий путь публичным
    const isPublicPath = publicPaths.some(publicPath =>
        path.startsWith(publicPath)
    );
    
    // Если путь публичный, middleware пропускает запрос дальше
    if (isPublicPath) {
        return NextResponse.next();
    }
    
    // --- Логика защиты для НЕ-публичных путей (например, /schedule, /top) ---
    const sessionCookie = request.cookies.get('auth-session');

    if (!sessionCookie) {
        // Если путь защищенный и нет cookie сессии, редирект на главную
        return NextResponse.redirect(new URL('/', request.url));
    }
    
    // Если сессия есть, пропускаем
    return NextResponse.next();
}

// Конфигурация matcher остается. Она эффективно применяет middleware ко всем запросам,
// кроме статических файлов.
export const config = {
    matcher: [
        '/((?!_next/static|_next/image|favicon.ico).*)',
    ],
};