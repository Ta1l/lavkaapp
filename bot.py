import logging
import asyncio
import os
import re
import requests
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

# === Конфиг ===
API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"
WEBAPP_URL = "https://slotworker.ru"

# === Логирование ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# === Состояния ===
class AuthState(StatesGroup):
    waiting_for_login = State()
    waiting_for_password = State()

class SlotState(StatesGroup):
    waiting_for_slots = State()

# === API Функции ===
async def get_api_key(username: str, password: str) -> str | None:
    try:
        r = requests.post(
            f"{WEBAPP_URL}/api/auth/get-token",
            json={"username": username, "password": password},
            timeout=10
        )
        if r.status_code == 200:
            api_key = r.json().get("apiKey")
            logger.info(f"Получен apiKey для {username}")
            return api_key
        else:
            logger.warning(f"Ошибка API аутентификации {r.status_code}: {r.text}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при запросе токена: {e}")
        return None


async def add_shift(api_key: str, date: str, start: str, end: str) -> bool:
    try:
        r = requests.post(
            f"{WEBAPP_URL}/api/shifts",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "date": date,
                "startTime": start,
                "endTime": end,
                "assignToSelf": True
            },
            timeout=10
        )
        if r.status_code in (200, 201):
            logger.info(f"Слот {date} {start}-{end} добавлен")
            return True
        else:
            logger.warning(f"Ошибка API добавления слота {r.status_code}: {r.text}")
            return False
    except Exception as e:
        logger.error(f"Ошибка при добавлении слота: {e}")
        return False

# === Вспомогательная функция парсинга ===
def parse_slot_input(text: str):
    text = text.lower().strip()
    today = datetime.now().date()
    if text.startswith("сегодня"):
        date = today
        time_part = text.replace("сегодня", "").strip()
    elif text.startswith("завтра"):
        date = today + timedelta(days=1)
        time_part = text.replace("завтра", "").strip()
    else:
        match = re.match(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}:\d{2})-(\d{2}:\d{2})", text)
        if not match:
            return None
        d, m, y, start, end = match.groups()
        date = datetime(int(y), int(m), int(d)).date()
        return {"date": date.strftime("%Y-%m-%d"), "start": start, "end": end}

    match = re.match(r"(\d{2}:\d{2})-(\d{2}:\d{2})", time_part)
    if not match:
        return None
    start, end = match.groups()
    return {"date": date.strftime("%Y-%m-%d"), "start": start, "end": end}

# === Хэндлеры ===
@dp.message(Command("start"))
async def start_cmd(message: Message, state: FSMContext):
    await message.answer("Введите логин:")
    await state.set_state(AuthState.waiting_for_login)


@dp.message(StateFilter(AuthState.waiting_for_login))
async def login_input(message: Message, state: FSMContext):
    await state.update_data(login=message.text.strip())
    await message.answer("Введите пароль:")
    await state.set_state(AuthState.waiting_for_password)


@dp.message(StateFilter(AuthState.waiting_for_password))
async def password_input(message: Message, state: FSMContext):
    data = await state.get_data()
    login = data.get("login")
    password = message.text.strip()

    api_key = await get_api_key(login, password)
    if api_key:
        await state.update_data(api_key=api_key)
        await state.set_state(SlotState.waiting_for_slots)
        await message.answer("✅ Вход успешен!\nТеперь отправьте слот в формате:\n`26.08.2025 09:00-17:00`\nили `сегодня 10:00-15:00`", parse_mode="Markdown")
    else:
        await message.answer("❌ Ошибка входа. Попробуйте снова: /start")
        await state.clear()


@dp.message(StateFilter(SlotState.waiting_for_slots))
async def add_slot_handler(message: Message, state: FSMContext):
    data = await state.get_data()
    api_key = data.get("api_key")

    slot = parse_slot_input(message.text)
    if not slot:
        await message.answer("⚠️ Неверный формат. Пример: `31.12.2025 20:00-23:00`", parse_mode="Markdown")
        return

    ok = await add_shift(api_key, slot["date"], slot["start"], slot["end"])
    if ok:
        await message.answer(f"✅ Слот добавлен: {slot['date']} {slot['start']}-{slot['end']}")
    else:
        await message.answer("❌ Ошибка при добавлении слота. Проверьте логи.")


# === MAIN ===
async def main():
    logger.info("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
