// src/lib/redis.ts
import { Redis } from 'ioredis';

let redis: Redis | null = null;

export function getRedisClient(): Redis {
  if (!redis) {
    const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
    redis = new Redis(REDIS_URL, {
      lazyConnect: true,
      maxRetriesPerRequest: 3,
    });
    redis.on('connect', () => console.log('[Redis] Подключение установлено'));
    redis.on('error', (err) => console.error('[Redis Error]', err));
  }
  return redis;
}
