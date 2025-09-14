// scripts/add-api-key-column.ts
import { pool } from '../src/lib/db';

async function migrate() {
  try {
    // Проверяем, существует ли колонка
    const checkColumn = await pool.query(`
      SELECT column_name 
      FROM information_schema.columns 
      WHERE table_name = 'users' AND column_name = 'api_key'
    `);

    if (checkColumn.rows.length === 0) {
      // Добавляем колонку
      await pool.query('ALTER TABLE users ADD COLUMN api_key VARCHAR(255)');
      console.log('✅ Колонка api_key успешно добавлена');
      
      // Создаем индекс
      await pool.query('CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)');
      console.log('✅ Индекс для api_key создан');
    } else {
      console.log('ℹ️ Колонка api_key уже существует');
    }

    // Проверяем структуру таблицы
    const structure = await pool.query(`
      SELECT column_name, data_type, is_nullable
      FROM information_schema.columns 
      WHERE table_name = 'users'
      ORDER BY ordinal_position
    `);
    
    console.log('\n📋 Структура таблицы users:');
    structure.rows.forEach(col => {
      console.log(`  - ${col.column_name}: ${col.data_type} ${col.is_nullable === 'NO' ? 'NOT NULL' : ''}`);
    });

  } catch (error) {
    console.error('❌ Ошибка миграции:', error);
  } finally {
    await pool.end();
  }
}

migrate();