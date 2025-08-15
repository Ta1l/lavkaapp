// src/lib/db.ts
import { Pool } from 'pg';

// Для продакшена используем переменные окружения, для разработки - дефолтные значения
const dbConfig = {
  user: process.env.POSTGRES_USER || 'lavka_user',
  host: process.env.POSTGRES_HOST || 'localhost',
  database: process.env.POSTGRES_DATABASE || 'schedule_db',
  password: process.env.POSTGRES_PASSWORD || 'hw6uxxs9*Hz5',
  port: Number(process.env.POSTGRES_PORT || 5432),
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
  ssl: process.env.POSTGRES_SSL === 'true' ? { rejectUnauthorized: false } : undefined,
};

// Логирование конфигурации (без пароля)
console.log('[DB] Конфигурация подключения:', {
  ...dbConfig,
  password: '***hidden***'
});

export const pool = new Pool(dbConfig);

pool.on('connect', () => {
  console.log('[DB] Пул успешно подключился к базе данных');
});

pool.on('error', (err) => {
  console.error('[DB Pool Error] Ошибка пула соединений:', err);
});

// Тестовое подключение при запуске
pool.query('SELECT NOW()', (err, res) => {
  if (err) {
    console.error('[DB] Ошибка тестового подключения:', err);
  } else {
    console.log('[DB] Тестовое подключение успешно:', res.rows[0]);
  }
});