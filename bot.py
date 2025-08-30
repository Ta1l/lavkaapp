# bot.py

import logging
import asyncio
import aiohttp
import re
import os
import json
from datetime import datetime, timedelta
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

from ocr_module import SlotParser  # вынес OCR в отдельный файл ocr_module.py

# === Конфиг ===
API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"
WEBAPP_URL = "https://slotworker.ru"
UPLOAD_DIR = "./user_uploads"

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
    waiting_for_screens = State()
    confirm_slots = State()

# === API Функции ===
async def get_api_key(username: str, password: str) -> str | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/auth/get-token",
                json={"username": username, "password": password},
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("apiKey")
                else:
                    return None
    except Exception as e:
        logger.error(f"Ошибка при запросе токена: {e}")
        return None

async def add_shift(api_key: str, date: str, start: str, end: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "date": date,
                    "startTime": start,
                    "endTime": end,
                    "assignToSelf": True
                },
                timeout=10
            ) as response:
                return response.status in (200, 201)
    except Exception as e:
        logger.error(f"Ошибка при добавлении слота: {e}")
        return False

# === OCR интеграция ===
def extract_slots_from_files(user_dir: str):
    parser = SlotParser(base_path=user_dir)
    slots = parser.process_all_screenshots()
    result = []
    for s in slots:
        result.append({
            "date": s["date"],
            "start": s["start_time"],
            "end": s["end_time"]
        })
    return result

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
        await message.answer("✅ Вход успешен! Отправьте скриншоты с расписанием.")
        await state.set_state(SlotState.waiting_for_screens)
    else:
        await message.answer("❌ Ошибка входа. Попробуйте снова: /start")
        await state.clear()

@dp.message(StateFilter(SlotState.waiting_for_screens), F.photo)
async def handle_screenshot(message: Message, state: FSMContext):
    data = await state.get_data()
    user_id = message.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    photo = message.photo[-1]
    file_path = os.path.join(user_dir, f"{photo.file_unique_id}.jpg")
    await photo.download(destination_file=file_path)

    await message.answer("📷 Скрин сохранён. Отправьте ещё или напишите /done когда закончите.")

@dp.message(StateFilter(SlotState.waiting_for_screens), Command("done"))
async def process_screens(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))

    if not os.path.exists(user_dir):
        await message.answer("⚠️ Вы не отправили скринов.")
        return

    slots = extract_slots_from_files(user_dir)
    if not slots:
        await message.answer("❌ Слоты не распознаны.")
        return

    await state.update_data(slots=slots)

    text = "Найдены следующие слоты:\n"
    for s in slots:
        text += f"- {s['date']} {s['start']}-{s['end']}\n"

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Да", callback_data="confirm_yes"),
             InlineKeyboardButton(text="❌ Нет", callback_data="confirm_no")]
        ]
    )

    await message.answer(text, reply_markup=kb)
    await state.set_state(SlotState.confirm_slots)

@dp.callback_query(StateFilter(SlotState.confirm_slots), F.data == "confirm_yes")
async def confirm_yes(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    api_key = data.get("api_key")
    slots = data.get("slots", [])

    ok_count = 0
    for s in slots:
        if await add_shift(api_key, s["date"], s["start"], s["end"]):
            ok_count += 1

    await call.message.answer(f"✅ Загружено {ok_count} слотов в расписание.")
    await state.clear()

@dp.callback_query(StateFilter(SlotState.confirm_slots), F.data == "confirm_no")
async def confirm_no(call: CallbackQuery, state: FSMContext):
    await call.message.answer("❌ Отменено. Начните заново: /start")
    await state.clear()

# === MAIN ===
async def main():
    logger.info("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
