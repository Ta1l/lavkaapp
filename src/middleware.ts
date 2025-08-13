// src/middleware.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const publicPaths = [
  '/',
  '/api/auth',
  '/favicon.ico',
];

export function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;

  const isPublicPath = publicPaths.some(publicPath =>
    path === publicPath ||
    (publicPath.startsWith('/api/') && path.startsWith(publicPath))
  );

  if (isPublicPath) {
    return NextResponse.next();
  }

  const sessionCookie = request.cookies.get('auth-session');

  if (!sessionCookie) {
    const url = request.nextUrl.clone();
    url.pathname = '/';
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
