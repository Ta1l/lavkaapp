// utils/weather.js
// Утилита для получения и автообновления погоды (Open-Meteo).
// Обновление по умолчанию: каждые 3 часа.
// Возвращает подписываемым компонентам строку вида: "Пасмурно, 3 градуса"

const DEFAULT_LAT = 60.031027;
const DEFAULT_LON = 30.324106;
const THREE_HOURS_MS = 3 * 60 * 60 * 1000;
const DEFAULT_TTL_MS = 10 * 60 * 1000; // 10 минут кеш внутри fetch

let latestData = null; // нормализованный ответ от API
let latestFormatted = null; // строка "Пасмурно, 3 градуса"
const subscribers = new Set(); // callback: (formattedString) => void

let autoTimer = null;

/* === Вспомогательные функции === */

function buildUrl(lat, lon, opts = {}) {
  const hourly = (opts.hourly || [
    "temperature_2m",
    "precipitation",
    "weathercode",
    "windspeed_10m",
  ]).join(",");
  const forecast_days = opts.forecast_days ?? 1;
  const timezone = opts.timezone ?? "auto";
  const extra = opts.query ? "&" + new URLSearchParams(opts.query).toString() : "";
  return `https://api.open-meteo.com/v1/forecast?latitude=${encodeURIComponent(
    lat
  )}&longitude=${encodeURIComponent(lon)}&hourly=${encodeURIComponent(
    hourly
  )}&current_weather=true&forecast_days=${forecast_days}&timezone=${encodeURIComponent(
    timezone
  )}${extra}`;
}

// Простая мапа weathercode -> короткое русское описание
function weatherCodeToRussian(code) {
  // codes: https://open-meteo.com/en/docs#api_form
  switch (code) {
    case 0: return "Ясно";
    case 1: return "Малооблачно";
    case 2: return "Облачно";
    case 3: return "Пасмурно";
    case 45:
    case 48: return "Туман";
    case 51:
    case 53:
    case 55: return "Морось";
    case 56:
    case 57: return "Лёгкий дождь (замерзающий)";
    case 61:
    case 63:
    case 65: return "Дождь";
    case 66:
    case 67: return "Сильный дождь (замерзающий)";
    case 71:
    case 73:
    case 75: return "Снег";
    case 77: return "Снежная крупа";
    case 80:
    case 81:
    case 82: return "Ливни";
    case 85:
    case 86: return "Снежные ливни";
    case 95: return "Гроза";
    case 96:
    case 99: return "Гроза с градом";
    default: return "Неизв.";
  }
}

function degreeDeclension(n) {
  // русское склонение "градус/градуса/градусов"
  const abs = Math.abs(Math.round(n));
  const lastTwo = abs % 100;
  if (lastTwo >= 11 && lastTwo <= 19) return "градусов";
  const last = abs % 10;
  if (last === 1) return "градус";
  if (last >= 2 && last <= 4) return "градуса";
  return "градусов";
}

function formatWeatherText(normalized) {
  if (!normalized) return null;
  const cur = normalized.current;
  if (cur && typeof cur.temperature !== "undefined") {
    const temp = Math.round(cur.temperature);
    const code = typeof cur.weathercode !== "undefined" ? cur.weathercode : null;
    const desc = code !== null ? weatherCodeToRussian(code) : "Погода";
    return `${desc}, ${temp} ${degreeDeclension(temp)}`;
  }

  // fallback: используем первый hourly timestamp если есть
  if (normalized.hourly && normalized.hourly.time && normalized.hourly.temperature_2m) {
    const t = normalized.hourly.temperature_2m[0];
    const wc = normalized.hourly.weathercode ? normalized.hourly.weathercode[0] : null;
    const desc = wc !== null ? weatherCodeToRussian(wc) : "Погода";
    const temp = Math.round(t);
    return `${desc}, ${temp} ${degreeDeclension(temp)}`;
  }

  return null;
}

/* === Fetch / нормализация === */

async function fetchWeatherNetwork(lat = DEFAULT_LAT, lon = DEFAULT_LON, opts = {}) {
  const url = buildUrl(lat, lon, opts);
  const res = await fetch(url);
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Weather API returned ${res.status}: ${txt}`);
  }
  const json = await res.json();
  // normalized: keep current_weather and hourly
  const normalized = {
    current: json.current_weather ?? null,
    hourly: json.hourly ?? null,
    raw: json,
    fetchedAt: Date.now(),
  };
  return normalized;
}

/* === Публичный API === */

export async function updateNow(lat = DEFAULT_LAT, lon = DEFAULT_LON, options = {}) {
  try {
    const data = await fetchWeatherNetwork(lat, lon, options.fetchOpts || {});
    latestData = data;
    latestFormatted = formatWeatherText(data);
    // оповестим подписчиков
    for (const cb of subscribers) {
      try { cb(latestFormatted); } catch (e) { console.error("subscriber error", e); }
    }
    return latestFormatted;
  } catch (err) {
    console.error("weather.updateNow error:", err);
    throw err;
  }
}

/**
 * Возвращает последнюю строку прогноза или null
 */
export function getLatestWeatherText() {
  return latestFormatted;
}

/**
 * Подписка: callback receives formatted string (or null)
 * Возвращает функцию отписки
 */
export function subscribe(cb) {
  subscribers.add(cb);
  // сразу отправим текущее значение
  try { cb(latestFormatted); } catch (e) { /* ignore */ }
  return () => subscribers.delete(cb);
}

export function unsubscribe(cb) {
  subscribers.delete(cb);
}

/**
 * Запуск автo-обновления (стартует интервал, делает первый fetch)
 * Возвращает объект { stop }
 */
export function startAutoRefresh(lat = DEFAULT_LAT, lon = DEFAULT_LON, intervalMs = THREE_HOURS_MS, options = {}) {
  // если уже запущен — остановим и перезапустим
  stopAutoRefresh();

  // делаем сразу первый запрос (не блокируя)
  updateNow(lat, lon, options).catch((e) => {
    // лог ошибки, но продолжаем
    console.error("Initial weather fetch failed:", e);
  });

  autoTimer = setInterval(() => {
    updateNow(lat, lon, options).catch((e) => {
      console.error("Periodic weather fetch failed:", e);
    });
  }, intervalMs);

  return {
    stop: () => stopAutoRefresh(),
  };
}

export function stopAutoRefresh() {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
}

/* === Автозапуск: стартуем с дефолтными координатами и интервалом 3 часа === */
try {
  // если среда поддерживает fetch — стартуем автопроцесс
  // не блокируем импорт
  startAutoRefresh(DEFAULT_LAT, DEFAULT_LON, THREE_HOURS_MS);
} catch (e) {
  // игнорируем ошибки на старте
  console.warn("weather auto-start failed:", e);
}

/* Экспорт по умолчанию для удобства */
export default {
  updateNow,
  getLatestWeatherText,
  subscribe,
  unsubscribe,
  startAutoRefresh,
  stopAutoRefresh,
};
