// src/app/page.tsx
'use client';

import { useState } from 'react';

export default function AuthPage() {
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  // т.к. используем обычную форму, JS здесь минимален
  // Ошибки сервер вернёт JSON-телом только при fetch,
  // при submit формы он редиректит. Для простоты, ошибок тут не ловим.
  // Если нужно — можно сделать AJAX логин с fetch, но тогда вернётся к гонке cookie.

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-black p-8">
      <div className="w-full max-w-sm rounded-lg bg-[#1C1C1C] p-8 shadow-lg">
        <h1 className="mb-6 text-center text-3xl font-bold text-white">Лавка</h1>

        <form
          action="/api/auth"
          method="POST"
          className="space-y-6"
          onSubmit={() => setPending(true)}
        >
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-300">
              Имя пользователя
            </label>
            <input
              id="username"
              name="username"
              type="text"
              required
              className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50 px-3 py-2"
              autoComplete="username"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-300">
              Пароль
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50 px-3 py-2"
              autoComplete="current-password"
            />
          </div>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <div className="flex flex-col gap-4 pt-2">
            <button
              type="submit"
              name="action"
              value="login"
              disabled={pending}
              className="flex w-full justify-center rounded-md border border-transparent bg-[#ffed23] px-4 py-2 text-sm font-medium text-black shadow-sm hover:bg-yellow-400 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {pending ? 'Вход…' : 'Войти'}
            </button>

            <button
              type="submit"
              name="action"
              value="register"
              disabled={pending}
              className="flex w-full justify-center rounded-md border border-gray-600 bg-gray-700 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {pending ? 'Регистрация…' : 'Зарегистрироваться'}
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}
