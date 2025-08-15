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
        console.clear(); // Очищаем консоль для чистоты эксперимента
        console.log(`[CLIENT LOG] Начинаем отправку. Действие: "${action}", Имя: "${username}"`);

        try {
            const response = await fetch('/api/auth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password, action }),
            });

            console.log('[CLIENT LOG] Получен ответ от сервера. Статус:', response.status);
            console.log('[CLIENT LOG] response.ok:', response.ok);

            const data = await response.json();
            console.log('[CLIENT LOG] Тело ответа (data):', data);

            if (!response.ok) {
                console.log('[CLIENT LOG] Ответ НЕ успешный (status не 2xx). Выбрасываем ошибку.');
                throw new Error(data.error || 'Произошла ошибка');
            }

            console.log('[CLIENT LOG] Ответ успешный. Перенаправление на /schedule/0...');
            router.push('/schedule/0');
            router.refresh();

        } catch (err: any) {
            console.error('[CLIENT LOG] Ошибка в блоке try-catch:', err.message);
            setError(err.message);
        } finally {
            console.log('[CLIENT LOG] Блок finally. Снимаем флаг загрузки.');
            setIsLoading(false);
        }
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-center bg-black p-8">
            <div className="w-full max-w-sm rounded-lg bg-[#1C1C1C] p-8 shadow-lg">
                <h1 className="mb-6 text-center text-3xl font-bold text-white">Лавка</h1>
                <div className="space-y-6">
                    {/* ... остальной JSX без изменений ... */}
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
                            className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50"
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
                            className="mt-1 block w-full rounded-md border-gray-600 bg-gray-800 text-white shadow-sm focus:border-yellow-400 focus:ring focus:ring-yellow-300 focus:ring-opacity-50"
                        />
                    </div>
                    {error && <p className="text-sm text-red-500">{error}</p>}
                    <div className="flex flex-col gap-4 pt-2">
                        <button
                            type="button"
                            onClick={() => handleSubmit('login')}
                            disabled={isLoading || !username || !password}
                            className="flex w-full justify-center rounded-md border border-transparent bg-[#ffed23] px-4 py-2 text-sm font-medium text-black shadow-sm hover:bg-yellow-400 focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50"
                        >
                            {isLoading ? 'Вход...' : 'Войти'}
                        </button>
                        <button
                            type="button"
                            onClick={() => handleSubmit('register')}
                            disabled={isLoading || !username || !password}
                            className="flex w-full justify-center rounded-md border border-gray-600 bg-gray-700 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50"
                        >
                            {isLoading ? 'Регистрация...' : 'Зарегистрироваться'}
                        </button>
                    </div>
                </div>
            </div>
        </main>
    );
}