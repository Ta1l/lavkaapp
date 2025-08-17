// src/app/page.tsx
export default function AuthPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-black p-8">
      <div className="w-full max-w-sm rounded-lg bg-[#1C1C1C] p-8 shadow-lg">
        <h1 className="mb-6 text-center text-3xl font-bold text-white">Лавка</h1>

        <form action="/api/auth" method="POST" className="space-y-6">
          <div>
            <label
              htmlFor="username"
              className="block text-sm font-medium text-gray-300"
            >
              Имя пользователя
            </label>
            <input
              id="username"
              name="username"
              type="text"
              required
              autoComplete="username"
              className="mt-1 block w-full rounded-md bg-gray-800 text-white px-3 py-2 focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-300"
            >
              Пароль
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              autoComplete="current-password"
              className="mt-1 block w-full rounded-md bg-gray-800 text-white px-3 py-2 focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50"
            />
          </div>

          <div className="flex flex-col gap-4 pt-2">
            <button
              type="submit"
              name="action"
              value="login"
              className="w-full rounded-md bg-yellow-400 px-4 py-2 text-sm font-medium text-black hover:bg-yellow-300 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 focus:ring-offset-gray-800"
            >
              Войти
            </button>

            <button
              type="submit"
              name="action"
              value="register"
              className="w-full rounded-md bg-gray-700 px-4 py-2 text-sm font-medium text-white hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-800"
            >
              Зарегистрироваться
            </button>
          </div>
        </form>
      </div>
    </main>
  );
}
