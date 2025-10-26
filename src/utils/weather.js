// utils/weather.js
// Получаем прогноз на сегодня + 5 дней (итого 6 дней) от Open-Meteo,
// формируем строки вида "Пасмурно, 3 градуса" для каждой даты YYYY-MM-DD,
// кэшируем и уведомляем подписчиков. Авто-обновление — каждые 3 часа.

const DEFAULT_LAT = 60.031027;
const DEFAULT_LON = 30.324106;
const THREE_HOURS_MS = 3 * 60 * 60 * 1000;
const DAYS_AHEAD = 5; // помимо сегодняшнего -> итого 6 дней (today + 5)
const DEFAULT_FORECAST_DAYS = 1 + DAYS_AHEAD; // API param

let latestDailyMap = {}; // { "YYYY-MM-DD": "Пасмурно, 3 градуса", ... }
let latestRaw = null;
const subscribers = new Set();
let autoTimer = null;

/* --- Helpers --- */

function buildUrl(lat, lon, opts = {}) {
  // запрашиваем daily: weathercode, temperature_2m_max, temperature_2m_min
  const forecast_days = opts.forecast_days ?? DEFAULT_FORECAST_DAYS;
  const timezone = opts.timezone ?? "auto";
  // daily params
  const daily = ["weathercode", "temperature_2m_max", "temperature_2m_min"].join(",");
  const extra = opts.query ? "&" + new URLSearchParams(opts.query).toString() : "";
  return `https://api.open-meteo.com/v1/forecast?latitude=${encodeURIComponent(
    lat
  )}&longitude=${encodeURIComponent(lon)}&daily=${encodeURIComponent(
    daily
  )}&current_weather=true&forecast_days=${forecast_days}&timezone=${encodeURIComponent(
    timezone
  )}${extra}`;
}

function weatherCodeToRussian(code) {
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
    case 57: return "Лёгкий дождь";
    case 61:
    case 63:
    case 65: return "Дождь";
    case 66:
    case 67: return "Сильный дождь";
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
  const abs = Math.abs(Math.round(n));
  const lastTwo = abs % 100;
  if (lastTwo >= 11 && lastTwo <= 19) return "градусов";
  const last = abs % 10;
  if (last === 1) return "градус";
  if (last >= 2 && last <= 4) return "градуса";
  return "градусов";
}

function formatDailyToMap(json) {
  // json.daily: { time: [...], weathercode: [...], temperature_2m_max: [...], temperature_2m_min: [...] }
  const map = {};
  if (!json || !json.daily || !Array.isArray(json.daily.time)) return map;

  const times = json.daily.time || [];
  const codes = json.daily.weathercode || [];
  const tmax = json.daily.temperature_2m_max || [];
  const tmin = json.daily.temperature_2m_min || [];

  for (let i = 0; i < times.length; i++) {
    const date = times[i]; // ISO date YYYY-MM-DD (Open-Meteo)
    const code = typeof codes[i] !== "undefined" ? codes[i] : null;
    const max = typeof tmax[i] !== "undefined" ? tmax[i] : null;
    const min = typeof tmin[i] !== "undefined" ? tmin[i] : null;
    // choose representative temperature: average of min and max if both present, else prefer max, then min
    let temp = null;
    if (max !== null && min !== null) temp = Math.round((max + min) / 2);
    else if (max !== null) temp = Math.round(max);
    else if (min !== null) temp = Math.round(min);
    const desc = code !== null ? weatherCodeToRussian(code) : "Погода";
    const formatted = (temp === null) ? `${desc}` : `${desc}, ${temp} ${degreeDeclension(temp)}`;
    map[date] = formatted;
  }
  return map;
}

/* --- Network fetch --- */

async function fetchForecast(lat = DEFAULT_LAT, lon = DEFAULT_LON, opts = {}) {
  const url = buildUrl(lat, lon, opts);
  const res = await fetch(url);
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Weather API returned ${res.status}: ${txt}`);
  }
  const json = await res.json();
  return json;
}

/* --- Public API --- */

/**
 * Выполнить немедленное обновление (fetch), заполнить внутренний map и уведомить подписчиков.
 * Возвращает объект { dailyMap, raw }
 */
export async function updateNow(lat = DEFAULT_LAT, lon = DEFAULT_LON, options = {}) {
  const json = await fetchForecast(lat, lon, { ...options.fetchOpts, forecast_days: DEFAULT_FORECAST_DAYS });
  latestRaw = json;
  latestDailyMap = formatDailyToMap(json);
  // notify subscribers with the whole map
  for (const cb of subscribers) {
    try { cb(latestDailyMap); } catch (e) { console.error("weather subscriber error", e); }
  }
  return { dailyMap: latestDailyMap, raw: latestRaw };
}

/**
 * Получить строку прогноза для даты ISO YYYY-MM-DD (синхронно).
 * Если нет данных — вернёт null.
 */
export function getWeatherForDate(isoDate) {
  if (!isoDate) return null;
  return latestDailyMap[isoDate] ?? null;
}

/**
 * Подписаться на обновления: callback receives the whole dailyMap (object).
 * Возвращает функцию отписки.
 */
export function subscribe(cb) {
  subscribers.add(cb);
  // сразу отправим текущее значение (если есть)
  try { cb(latestDailyMap); } catch (e) { /* ignore */ }
  return () => subscribers.delete(cb);
}

export function unsubscribe(cb) {
  subscribers.delete(cb);
}

/**
 * Автообновление: старт/стоп.
 * startAutoRefresh запускает немедленный fetch и затем интервал (по умолчанию 3 часа).
 * Вернёт объект { stop }
 */
export function startAutoRefresh(lat = DEFAULT_LAT, lon = DEFAULT_LON, intervalMs = THREE_HOURS_MS, options = {}) {
  stopAutoRefresh();
  // do immediate fetch (non-blocking)
  updateNow(lat, lon, options).catch((e) => console.error("Initial weather fetch failed:", e));
  autoTimer = setInterval(() => {
    updateNow(lat, lon, options).catch((e) => console.error("Periodic weather fetch failed:", e));
  }, intervalMs);
  return { stop: stopAutoRefresh };
}

export function stopAutoRefresh() {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
  }
}

/* --- Auto start --- */
try {
  // стартуем автoобновление по дефолту для координат проекта
  startAutoRefresh(DEFAULT_LAT, DEFAULT_LON, THREE_HOURS_MS);
} catch (e) {
  console.warn("weather auto-start failed:", e);
}

/* --- Default export --- */
export default {
  updateNow,
  getWeatherForDate,
  subscribe,
  unsubscribe,
  startAutoRefresh,
  stopAutoRefresh,
};
