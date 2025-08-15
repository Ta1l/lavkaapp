// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const publicPaths = new Set<string>([
  '/',              // страница логина/регистрации
  '/api/auth',      // обработчик логина/регистрации (любой метод)
  '/api/auth/logout',
  '/favicon.ico',
]);

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Пропускаем статические ассеты
  if (
    pathname.startsWith('/_next/static') ||
    pathname.startsWith('/_next/image') ||
    pathname === '/favicon.ico'
  ) {
    return NextResponse.next();
  }

  // Публичные пути
  const isPublic =
    pathname === '/' ||
    pathname === '/api/auth' ||
    pathname === '/api/auth/logout' ||
    pathname === '/favicon.ico';

  const sessionCookie = request.cookies.get('auth-session');

  // Авторизованный пользователь не должен попадать на страницу входа
  if (isPublic && pathname === '/' && sessionCookie) {
    return NextResponse.redirect(new URL('/schedule/0', request.url));
  }

  // Для всех остальных путей — обязательна сессия
  if (!isPublic && !sessionCookie) {
    const url = request.nextUrl.clone();
    url.pathname = '/';
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

// Матчер — всё, кроме статических файлов
export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
