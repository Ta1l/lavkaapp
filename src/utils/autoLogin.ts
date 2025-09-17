export async function autoLogin(): Promise<string | null> {
  try {
    // Проверяем, что мы в браузере
    if (typeof window === 'undefined') {
      return null;
    }

    // Сначала проверяем localStorage
    const storedKey = localStorage.getItem("apiKey");
    if (storedKey) {
      return storedKey;
    }

    // Пытаемся получить Telegram WebApp
    const tg = (window as any).Telegram?.WebApp;
    const telegramId = tg?.initDataUnsafe?.user?.id;

    if (!telegramId) {
      return null;
    }

    // Запрашиваем API key по Telegram ID
    const res = await fetch("/api/auth/auto-login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ telegramId }),
    });

    if (!res.ok) {
      return null;
    }

    const data = await res.json();
    if (data.apiKey) {
      localStorage.setItem("apiKey", data.apiKey);
      return data.apiKey;
    }

    return null;
  } catch (error) {
    console.error("Ошибка автологина:", error);
    return null;
  }
}