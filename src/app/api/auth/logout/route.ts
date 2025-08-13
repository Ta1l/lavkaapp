// src/app/api/auth/logout/route.ts

import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST() {
  try {
    // Получаем объект cookies
    const cookieStore = cookies();
    
    // Удаляем cookie сессии
    cookieStore.set('auth-session', '', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 0, // Немедленное удаление
      path: '/',
    });

    return NextResponse.json({ message: 'Вы успешно вышли' }, { status: 200 });
  } catch (error) {
    console.error('[API Logout Error]', error);
    return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
  }
}