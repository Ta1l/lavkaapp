// src/lib/db.ts
import { Pool } from 'pg';

const requiredEnvVars = [
  'POSTGRES_USER',
  'POSTGRES_PASSWORD',
  'POSTGRES_HOST',
  'POSTGRES_PORT',
  'POSTGRES_DATABASE',
];

const missingVars = requiredEnvVars.filter(v => !process.env[v]?.trim());
if (missingVars.length > 0) {
  throw new Error(
    `Critical Error: Missing required DB env variables: ${missingVars.join(', ')}`
  );
}

export const pool = new Pool({
  user: process.env.POSTGRES_USER,
  host: process.env.POSTGRES_HOST,
  database: process.env.POSTGRES_DATABASE,
  password: process.env.POSTGRES_PASSWORD,
  port: Number(process.env.POSTGRES_PORT),
  max: 10,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
  ssl: process.env.POSTGRES_SSL === 'true' ? { rejectUnauthorized: false } : undefined,
});

pool.on('connect', () => {
  console.log('[DB] Пул успешно подключился к базе данных');
});

pool.on('error', (err) => {
  console.error('[DB Pool Error] Ошибка пула соединений:', err);
});
