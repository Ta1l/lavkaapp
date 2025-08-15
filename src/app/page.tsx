"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function AuthPage() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const router = useRouter();

    const handleSubmit = async (action: 'login' | 'register') => {
        setIsLoading(true);
        setError(null);

        try {
            console.log(`[КЛИЕНТ] Отправка ${action} запроса...`);
            
            const response = await fetch('/api/auth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password, action }),
                credentials: 'include', // Важно для работы с куками
            });

            const data = await response.json();
            console.log(`[КЛИЕНТ] Получен ответ:`, data);

            if (!response.ok) {
                throw new Error(data.error || 'Произошла ошибка');
            }

            // Успешная авторизация
            console.log(`[КЛИЕНТ] ${action} успешен, редирект...`);
            
            // Используем router.push вместо window.location для более плавного перехода
            router.push('/schedule/0');
            
            // Если router.push не сработает, используем запасной вариант
            setTimeout(() => {
                window.location.href = '/schedule/0';
            }, 100);

        } catch (err: any) {
            console.error(`[КЛИЕНТ] Ошибка:`, err);
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-center bg-black p-8">
            <div className="w-full max-w-sm rounded-lg bg-[#1C1C1C] p-8 shadow-lg">
                <h1 className="mb-6 text-center text-3xl font-bold text-white">Лавка</h1>
                
                <form onSubmit={(e) => e.preventDefault()} className="space-y-6">
                    <div>
                        <label htmlFor="username" className="block text-sm font-medium text-gray-300">
                            Имя пользователя
                        </label>
                        <input
                            id="username"
                            name="username"
                            type="text"
                            required
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50 px-3 py-2"
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
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50 px-3 py-2"
                        />
                    </div>

                    {error && <p className="text-sm text-red-500">{error}</p>}

                    <div className="flex flex-col gap-4 pt-2">
                        <button
                            type="button"
                            onClick={() => handleSubmit('login')}
                            disabled={isLoading || !username || !password}
                            className="flex w-full justify-center rounded-md border border-transparent bg-[#ffed23] px-4 py-2 text-sm font-medium text-black shadow-sm hover:bg-yellow-400 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isLoading ? 'Вход...' : 'Войти'}
                        </button>
                        <button
                            type="button"
                            onClick={() => handleSubmit('register')}
                            disabled={isLoading || !username || !password}
                            className="flex w-full justify-center rounded-md border border-gray-600 bg-gray-700 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isLoading ? 'Регистрация...' : 'Зарегистрироваться'}
                        </button>
                    </div>
                </form>
            </div>
        </main>
    );
}