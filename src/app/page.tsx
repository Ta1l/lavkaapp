// src/app/page.tsx
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

// Добавляем декларацию типа
declare global {
  interface Window {
    Telegram?: {
      WebApp?: {
        ready: () => void;
        expand: () => void;
        initDataUnsafe?: {
          user?: {
            id: number;
            first_name?: string;
            last_name?: string;
            username?: string;
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
  const [checkingTelegram, setCheckingTelegram] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // Инициализация Telegram WebApp
    if (window.Telegram?.WebApp) {
      window.Telegram.WebApp.ready();
      window.Telegram.WebApp.expand();
    }

    // Проверяем автоматический вход через Telegram
    checkAutoLogin();
  }, []);

  const checkAutoLogin = async () => {
    try {
      // Проверяем, открыто ли приложение через Telegram
      const telegramUser = window.Telegram?.WebApp?.initDataUnsafe?.user;
      
      if (!telegramUser?.id) {
        console.log('Not in Telegram context');
        setCheckingTelegram(false);
        checkUrlParams();
        return;
      }

      console.log('Telegram user detected:', telegramUser);

      // Пытаемся автоматически войти
      const response = await fetch("/api/auth/auto-login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ telegramId: String(telegramUser.id) })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.apiKey) {
          console.log('✅ Auto-login successful');
          localStorage.setItem("apiKey", data.apiKey);
          window.location.href = "/schedule/0";
          return;
        }
      } else if (response.status === 404) {
        // Пользователь не связан - показываем форму входа
        console.log('Telegram account not linked yet');
        setError('Для первого входа введите логин и пароль');
      }
    } catch (err) {
      console.error("Auto-login error:", err);
    } finally {
      setCheckingTelegram(false);
      checkUrlParams();
    }
  };

  const checkUrlParams = () => {
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
  };

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
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

      if (response.redirected) {
        const url = new URL(response.url);
        if (url.pathname === "/schedule/0") {
          // Успешный вход/регистрация - получаем API ключ
          try {
            const headers: HeadersInit = {
              "Content-Type": "application/json"
            };
            
            // Добавляем Telegram ID если есть
            if (window.Telegram?.WebApp?.initDataUnsafe?.user?.id) {
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
            }
          } catch (err) {
            console.error("Error getting token:", err);
          }
          
          window.location.href = "/schedule/0";
        } else {
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

  // Показываем индикатор загрузки при проверке Telegram
  if (checkingTelegram) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4">
        <div className="text-white">Загрузка...</div>
      </div>
    );
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
          
          {window.Telegram?.WebApp?.initDataUnsafe?.user && (
            <div className="bg-blue-500/20 border border-blue-500 text-blue-400 p-3 rounded text-sm">
              👤 Вход через Telegram: {window.Telegram.WebApp.initDataUnsafe.user.first_name}
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