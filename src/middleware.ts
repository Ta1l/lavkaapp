// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const publicPaths = [
  '/auth',           // Страница авторизации
  '/api/auth',       // API endpoint для авторизации
  '/favicon.ico',
];

export function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;
  
  // Проверяем, является ли путь публичным
  const isPublicPath = publicPaths.some(publicPath =>
    path === publicPath ||
    path.startsWith(publicPath + '/') ||
    (publicPath.startsWith('/api/') && path.startsWith(publicPath))
  );

  // Получаем сессию из cookies
  const sessionCookie = request.cookies.get('auth-session');
  
  // Логирование для отладки
  console.log(`[MIDDLEWARE] Path: ${path}, Has session: ${!!sessionCookie}, Is public: ${isPublicPath}`);

  // Если путь публичный
  if (isPublicPath) {
    // Если пользователь авторизован и пытается зайти на /auth, редиректим на главную
    if (sessionCookie && path === '/auth') {
      console.log('[MIDDLEWARE] Авторизованный пользователь на /auth, редирект на /schedule/0');
      return NextResponse.redirect(new URL('/schedule/0', request.url));
    }
    // В остальных случаях пропускаем
    return NextResponse.next();
  }

  // Для защищенных путей проверяем наличие сессии
  if (!sessionCookie) {
    console.log('[MIDDLEWARE] Нет сессии для защищенного пути, редирект на /auth');
    const url = request.nextUrl.clone();
    url.pathname = '/auth';
    return NextResponse.redirect(url);
  }

  // Проверяем валидность сессии (базовая проверка)
  try {
    const sessionData = JSON.parse(sessionCookie.value);
    if (!sessionData.id || !sessionData.username) {
      console.log('[MIDDLEWARE] Невалидная сессия, редирект на /auth');
      const url = request.nextUrl.clone();
      url.pathname = '/auth';
      return NextResponse.redirect(url);
    }
  } catch (error) {
    console.error('[MIDDLEWARE] Ошибка парсинга сессии:', error);
    const url = request.nextUrl.clone();
    url.pathname = '/auth';
    return NextResponse.redirect(url);
  }

  // Если корневой путь и пользователь авторизован, редиректим на расписание
  if (path === '/' && sessionCookie) {
    console.log('[MIDDLEWARE] Авторизованный пользователь на /, редирект на /schedule/0');
    return NextResponse.redirect(new URL('/schedule/0', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};