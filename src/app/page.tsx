"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function AuthPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const tg = (window as any).Telegram?.WebApp;

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>, action: "login" | "register") {
    e.preventDefault();
    setError(null);

    try {
      const telegramId = tg?.initDataUnsafe?.user?.id;

      // отправляем данные на сервер
      const res = await fetch("/api/auth/get-token", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(telegramId ? { "x-telegram-id": telegramId } : {}),
        },
        body: JSON.stringify({ username, password, action }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "Ошибка входа");
        return;
      }

      if (data.apiKey) {
        localStorage.setItem("apiKey", data.apiKey);
        console.log("✅ Авторизация успешна, apiKey сохранён");
        router.push("/schedule/0"); // переход к расписанию
      }
    } catch (err) {
      console.error("Ошибка при авторизации:", err);
      setError("Ошибка сервера");
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-black p-8">
      <div className="w-full max-w-sm rounded-lg bg-[#1C1C1C] p-8 shadow-lg">
        <h1 className="mb-6 text-center text-3xl font-bold text-white">Лавка</h1>

        <form className="space-y-6" onSubmit={(e) => handleSubmit(e, "login")}>
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-300">
              Имя пользователя
            </label>
            <input
              id="username"
              name="username"
              type="text"
              required
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="mt-1 block w-full rounded-md bg-gray-800 text-white px-3 py-2 focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50"
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
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded-md bg-gray-800 text-white px-3 py-2 focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50"
            />
          </div>

          {error && <p className="text-red-500 text-sm">{error}</p>}

          <div className="flex flex-col gap-4 pt-2">
            <button
              type="submit"
              className="w-full rounded-md bg-yellow-400 px-4 py-2 text-sm font-medium text-black hover:bg-yellow-300 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 focus:ring-offset-gray-800"
            >
              Войти
            </button>

            <button
              type="button"
              onClick={(e) => handleSubmit(e as any, "register")}
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
