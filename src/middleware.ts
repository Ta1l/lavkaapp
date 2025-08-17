// src/middleware.ts
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Публичные маршруты (без авторизации)
const isPublicPath = (pathname: string) =>
  pathname === "/" ||
  pathname === "/favicon.ico" ||
  pathname === "/api/auth" ||
  pathname === "/api/auth/logout";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Пропускаем статические ассеты
  if (
    pathname.startsWith("/_next/static") ||
    pathname.startsWith("/_next/image") ||
    pathname === "/favicon.ico"
  ) {
    return NextResponse.next();
  }

  const sessionCookie = request.cookies.get("auth-session");

  // Авторизованный пользователь не должен попадать на логин
  if (pathname === "/" && sessionCookie) {
    return NextResponse.redirect(new URL("/schedule/0", request.url));
  }

  // Для всех остальных путей — обязателен сеанс
  if (!isPublicPath(pathname) && !sessionCookie) {
    const url = request.nextUrl.clone();
    url.pathname = "/";
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

// Матчер — всё, кроме статических файлов
export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
