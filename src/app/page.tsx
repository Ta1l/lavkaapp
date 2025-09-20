// src/app/page.tsx
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

// Добавляем декларацию типа
declare global {
  interface Window {
    Telegram?: {
      WebApp?: {
        initDataUnsafe?: {
          user?: {
            id: number;
          };
        };
      };
    };
  }
}

export default function HomePage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLogin, setIsLogin] = useState(true);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // Проверяем параметры URL для отображения ошибок
    const params = new URLSearchParams(window.location.search);
    const errorParam = params.get('error');
    
    if (errorParam) {
      switch(errorParam) {
        case 'user_exists':
          setError('Пользователь с таким именем уже существует');
          break;
        case 'invalid_credentials':
          setError('Неверный логин или пароль');
          break;
        case 'validation_failed':
          setError('Заполните все поля');
          break;
        default:
          setError('Произошла ошибка');
      }
    }
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      // Сохраняем логин и пароль для последующего получения токена
      const savedUsername = username;
      const savedPassword = password;

      const formData = new FormData();
      formData.append("username", username);
      formData.append("password", password);
      formData.append("action", isLogin ? "login" : "register");

      const response = await fetch("/api/auth", {
        method: "POST",
        body: formData,
      });

      // Проверяем редирект
      if (response.redirected) {
        const url = new URL(response.url);
        if (url.pathname === "/schedule/0") {
          // Успешный вход/регистрация - теперь получаем API ключ
          try {
            const headers: HeadersInit = {
              "Content-Type": "application/json"
            };
            
            // Добавляем Telegram ID если есть
            if (typeof window !== 'undefined' && window.Telegram?.WebApp?.initDataUnsafe?.user?.id) {
              headers["x-telegram-id"] = String(window.Telegram.WebApp.initDataUnsafe.user.id);
            }

            const tokenRes = await fetch("/api/auth/get-token", {
              method: "POST",
              headers,
              body: JSON.stringify({
                username: savedUsername,
                password: savedPassword
              })
            });
            
            if (tokenRes.ok) {
              const data = await tokenRes.json();
              if (data.apiKey) {
                localStorage.setItem("apiKey", data.apiKey);
                console.log("✅ API ключ сохранен");
              }
            } else {
              console.error("Failed to get API token");
            }
          } catch (err) {
            console.error("Error getting token:", err);
          }
          
          // Переходим на страницу расписания
          window.location.href = "/schedule/0";
        } else {
          // Ошибка - парсим параметры
          const errorParam = url.searchParams.get('error');
          if (errorParam === 'user_exists') {
            setError('Пользователь с таким именем уже существует');
          } else if (errorParam === 'invalid_credentials') {
            setError('Неверный логин или пароль');
          } else {
            setError('Произошла ошибка');
          }
        }
      }
    } catch (err) {
      console.error("Submit error:", err);
      setError("Ошибка соединения с сервером");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <h1 className="text-3xl font-bold text-white text-center mb-8">
          SlotWorker
        </h1>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <h2 className="text-xl font-semibold text-white text-center mb-4">
            {isLogin ? "Вход" : "Регистрация"}
          </h2>
          
          {error && (
            <div className="bg-red-500/20 border border-red-500 text-red-500 p-3 rounded text-sm">
              {error}
            </div>
          )}
          
          <input
            type="text"
            placeholder="Логин"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full p-3 bg-gray-800 text-white rounded border border-gray-700 focus:border-yellow-400 focus:outline-none"
            required
            disabled={isLoading}
          />
          
          <input
            type="password"
            placeholder="Пароль"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-3 bg-gray-800 text-white rounded border border-gray-700 focus:border-yellow-400 focus:outline-none"
            required
            disabled={isLoading}
          />
          
          <button
            type="submit"
            disabled={isLoading}
            className="w-full p-3 bg-yellow-400 text-black font-semibold rounded hover:bg-yellow-500 disabled:opacity-50 transition-colors"
          >
            {isLoading ? "Загрузка..." : (isLogin ? "Войти" : "Зарегистрироваться")}
          </button>
          
          <button
            type="button"
            onClick={() => {
              setIsLogin(!isLogin);
              setError("");
            }}
            className="w-full text-gray-400 text-sm hover:text-white transition-colors"
            disabled={isLoading}
          >
            {isLogin ? "Нет аккаунта? Зарегистрироваться" : "Уже есть аккаунт? Войти"}
          </button>
        </form>
      </div>
    </div>
  );
}