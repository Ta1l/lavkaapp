const tg = (window as any).Telegram?.WebApp;

export async function autoLogin() {
  const telegramId = tg?.initDataUnsafe?.user?.id;
  if (!telegramId) return null;

  try {
    const res = await fetch("/api/auth/auto-login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ telegramId }),
    });

    if (res.ok) {
      const data = await res.json();
      return data.apiKey;
    }
    return null;
  } catch (err) {
    console.error("[autoLogin] Error:", err);
    return null;
  }
}