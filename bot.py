# bot.py
import asyncio
import logging
import os

# Импортируем ключевые компоненты из нашего пакета 'bot'
from bot import dp, bot, register_all_handlers, UPLOAD_DIR, logger

async def main():
    """Основная функция для запуска бота."""
    
    # Настраиваем логирование, чтобы видеть, что происходит
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    logger.info("🚀 Бот запускается...")

    # Убедимся, что папка для загрузок существует
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info(f"📤 Папка для загрузок: {os.path.abspath(UPLOAD_DIR)}")
    
    # Регистрируем все обработчики, описанные в bot/handlers/
    register_all_handlers() # 
    
    # Запускаем бота
    # bot.delete_webhook используется на случай, если бот был где-то запущен на вебхуках
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 Бот остановлен.")