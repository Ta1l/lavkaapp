// src/app/page.tsx
"use client";
import { useState } from 'react';

export default function AuthPage() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (action: 'login' | 'register') => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/auth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password, action }),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Произошла ошибка');
            }
            window.location.href = '/schedule/0';
        } catch (err: any) {
            setError(err.message);
            // После неудачной попытки очищаем поля
            setUsername('');
            setPassword('');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <main className="flex min-h-screen flex-col items-center justify-center bg-black p-8">
            <div className="w-full max-w-sm rounded-lg bg-[#1C1C1C] p-8 shadow-lg">
                <h1 className="mb-6 text-center text-3xl font-bold text-white">Лавка</h1>
                <form onSubmit={(e) => { e.preventDefault(); handleSubmit('login'); }} className="space-y-6">
                    <div>
                        <label htmlFor="username">Имя пользователя</label>
                        <input id="username" name="username" type="text" required value={username} onChange={(e) => setUsername(e.target.value)} />
                    </div>
                    <div>
                        <label htmlFor="password">Пароль</label>
                        <input id="password" name="password" type="password" required value={password} onChange={(e) => setPassword(e.target.value)} />
                    </div>
                    {error && <p className="text-sm text-red-500">{error}</p>}
                    <div className="flex flex-col gap-4 pt-2">
                        <button type="button" onClick={() => handleSubmit('login')} disabled={isLoading || !username || !password}>Войти</button>
                        <button type="button" onClick={() => handleSubmit('register')} disabled={isLoading || !username || !password}>Зарегистрироваться</button>
                    </div>
                </form>
            </div>
        </main>
    );
}