# bot/__init__.py
import os
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

# Конфиг через env (можно переопределить)
API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM")
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://slotworker.ru")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./user_uploads")

# Логирование
logger = logging.getLogger("lavka.bot")
logger.setLevel(logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

def register_all_handlers():
    """
    Импортируем и регистрируем обработчики. Вызывать при старте (один раз).
    """
    from .handlers import auth, slots
    auth.register_handlers(dp)
    slots.register_handlers(dp)


__all__ = ["bot", "dp", "register_all_handlers", "UPLOAD_DIR", "WEBAPP_URL"]