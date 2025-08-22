import asyncio
import asyncpg
import pytesseract
from PIL import Image
import re
import os

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import Command, StateFilter

API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"  # вставь свой токен

# ========= DB CONFIG =========
DB_CONFIG = {
    "user": "lavka_user",
    "password": "hw6uxxs9*Hz5",
    "database": "schedule_db",
    "host": "localhost",
    "port": 5432,
}

# ========= FSM STATES =========
class AuthState(StatesGroup):
    waiting_for_login = State()
    waiting_for_password = State()

class AddShiftState(StatesGroup):
    waiting_for_photos = State()
    waiting_for_confirmation = State()

# ========= OCR PARSER =========
def parse_slots(image_path: str):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="rus")

    slots = []
    current_day = None
    address = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # День недели
        if re.match(r"\d{1,2}\s+\w+", line.lower()):
            current_day = line
            continue

        # Адрес
        if "проспект" in line.lower() or "ул" in line.lower() or "д." in line.lower():
            address = line
            continue

        # Время
        time_match = re.search(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", line)
        if time_match:
            slot = {
                "day": current_day,
                "start_time": time_match.group(1),
                "end_time": time_match.group(2),
                "status": None,
                "address": address,
            }

            if "выполнен" in line.lower():
                slot["status"] = "выполнен"
            elif "отмен" in line.lower():
                slot["status"] = "отменён"
            elif "опоздан" in line.lower():
                slot["status"] = "выполнен с опозданием"
            else:
                slot["status"] = "неизвестно"

            slots.append(slot)

    return slots

# ========= BOT =========
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# хранение user_id после логина
user_sessions = {}

async def db_get_user(login, password):
    conn = await asyncpg.connect(**DB_CONFIG)
    row = await conn.fetchrow(
        "SELECT id FROM users WHERE login=$1 AND password=$2", login, password
    )
    await conn.close()
    return row["id"] if row else None

async def db_insert_shifts(user_id, slots):
    conn = await asyncpg.connect(**DB_CONFIG)
    for s in slots:
        await conn.execute(
            """
            INSERT INTO shifts(user_id, start_time, end_time, status, address)
            VALUES($1, $2, $3, $4, $5)
            """,
            user_id,
            s["start_time"],
            s["end_time"],
            s["status"],
            s["address"],
        )
    await conn.close()

# ========= COMMANDS =========
@dp.message(Command("start"))
async def start(message: Message, state: FSMContext):
    await message.answer("Введите ваш логин:")
    await state.set_state(AuthState.waiting_for_login)

@dp.message(StateFilter(AuthState.waiting_for_login))
async def get_login(message: Message, state: FSMContext):
    await state.update_data(login=message.text)
    await message.answer("Введите пароль:")
    await state.set_state(AuthState.waiting_for_password)

@dp.message(StateFilter(AuthState.waiting_for_password))
async def get_password(message: Message, state: FSMContext):
    data = await state.get_data()
    login = data["login"]
    password = message.text

    user_id = await db_get_user(login, password)
    if user_id:
        user_sessions[message.from_user.id] = user_id
        await message.answer("✅ Успешный вход! Теперь можете использовать /add для загрузки смен.")
        await state.clear()
    else:
        await message.answer("❌ Неверный логин или пароль. Попробуйте ещё раз.")
        await state.clear()

@dp.message(Command("add"))
async def add_shifts(message: Message, state: FSMContext):
    if message.from_user.id not in user_sessions:
        await message.answer("Сначала авторизуйтесь через /start")
        return

    await message.answer("Отправьте скриншоты смен:")
    await state.set_state(AddShiftState.waiting_for_photos)

@dp.message(StateFilter(AddShiftState.waiting_for_photos), F.photo)
async def handle_photos(message: Message, state: FSMContext):
    photo = message.photo[-1]
    file_path = f"/tmp/{photo.file_id}.jpg"
    await bot.download(photo, destination=file_path)

    slots = parse_slots(file_path)

    data = await state.get_data()
    all_slots = data.get("slots", [])
    all_slots.extend(slots)
    await state.update_data(slots=all_slots)

    await message.answer("📷 Скриншот обработан. Пришлите ещё или введите /done")

@dp.message(Command("done"), StateFilter(AddShiftState.waiting_for_photos))
async def confirm_slots(message: Message, state: FSMContext):
    data = await state.get_data()
    slots = data.get("slots", [])

    if not slots:
        await message.answer("❌ Нет обработанных слотов.")
        return

    text = "Найденные смены:\n"
    for s in slots:
        text += f"- {s['day']} {s['start_time']} - {s['end_time']} ({s['status']})\n"

    kb = InlineKeyboardBuilder()
    kb.button(text="Подтвердить ✅", callback_data="confirm")
    kb.button(text="Отмена ❌", callback_data="cancel")

    await message.answer(text, reply_markup=kb.as_markup())
    await state.set_state(AddShiftState.waiting_for_confirmation)

@dp.callback_query(F.data == "confirm", StateFilter(AddShiftState.waiting_for_confirmation))
async def save_slots(callback: CallbackQuery, state: FSMContext):
    user_id = user_sessions[callback.from_user.id]
    data = await state.get_data()
    slots = data["slots"]

    await db_insert_shifts(user_id, slots)
    await callback.message.answer("✅ Смены сохранены в базе.")
    await state.clear()

@dp.callback_query(F.data == "cancel", StateFilter(AddShiftState.waiting_for_confirmation))
async def cancel_slots(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("❌ Действие отменено.")
    await state.clear()

# ========= MAIN =========
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
