import logging
import asyncpg
import bcrypt
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import asyncio

from ocr_parser import parse_slots  # твой OCR парсер

# === Конфиг ===
API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"

DB_CONFIG = {
    "user": "lavka_user",
    "password": "hw6uxxs9*Hz5",
    "database": "schedule_db",
    "host": "localhost",
    "port": 5432,
}

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# === Состояния ===
class AuthState(StatesGroup):
    waiting_for_login = State()
    waiting_for_password = State()

class AddShiftState(StatesGroup):
    waiting_for_photos = State()
    confirming = State()


# === DB ===
async def db_get_user(login: str, password: str):
    conn = await asyncpg.connect(**DB_CONFIG)
    row = await conn.fetchrow(
        "SELECT id, password FROM users WHERE username=$1",
        login.strip()
    )
    await conn.close()

    if row and bcrypt.checkpw(password.strip().encode(), row["password"].encode()):
        return row["id"]
    return None


async def db_add_shift(user_id: int, date: str, time: str, status: str):
    conn = await asyncpg.connect(**DB_CONFIG)
    await conn.execute(
        """
        INSERT INTO shifts (user_id, shift_date, shift_time, status)
        VALUES ($1, $2, $3, $4)
        """,
        user_id, date, time, status
    )
    await conn.close()


# === START ===
@dp.message(Command("start"))
async def start(message: Message, state: FSMContext):
    await message.answer("Введите логин:")
    await state.set_state(AuthState.waiting_for_login)


# === LOGIN ===
@dp.message(StateFilter(AuthState.waiting_for_login))
async def get_login(message: Message, state: FSMContext):
    await state.update_data(login=message.text.strip())
    await message.answer("Введите пароль:")
    await state.set_state(AuthState.waiting_for_password)


# === PASSWORD ===
@dp.message(StateFilter(AuthState.waiting_for_password))
async def get_password(message: Message, state: FSMContext):
    data = await state.get_data()
    login = data.get("login")
    password = message.text.strip()

    user_id = await db_get_user(login, password)
    if user_id:
        await state.update_data(user_id=user_id)
        await message.answer(f"✅ Успешный вход! Добро пожаловать, {login}")
        await state.clear()
    else:
        await message.answer("❌ Неверный логин или пароль. Попробуйте снова: /start")
        await state.clear()


# === ADD SHIFTS ===
@dp.message(Command("add"))
async def add_shifts(message: Message, state: FSMContext):
    await message.answer("Отправьте скриншоты смен. Когда закончите, напишите /done")
    await state.set_state(AddShiftState.waiting_for_photos)


@dp.message(StateFilter(AddShiftState.waiting_for_photos), F.photo)
async def handle_photos(message: Message, state: FSMContext):
    file = await bot.get_file(message.photo[-1].file_id)
    file_path = f"downloads/{file.file_unique_id}.png"
    await bot.download_file(file.file_path, destination=file_path)

    data = await state.get_data()
    photos = data.get("photos", [])
    photos.append(file_path)
    await state.update_data(photos=photos)
    await message.answer("✅ Скриншот получен")


@dp.message(Command("done"), StateFilter(AddShiftState.waiting_for_photos))
async def confirm_slots(message: Message, state: FSMContext):
    data = await state.get_data()
    photos = data.get("photos", [])

    if not photos:
        await message.answer("Вы не отправили скриншоты")
        await state.clear()
        return

    all_slots = []
    for p in photos:
        slots = parse_slots(p)  # возвращает [{"date": ..., "time": ..., "status": ...}, ...]
        all_slots.extend(slots)

    if not all_slots:
        await message.answer("❌ Не удалось распознать слоты.")
        await state.clear()
        return

    await state.update_data(slots=all_slots)
    slots_text = "\n".join([f"{s['date']} {s['time']} [{s['status']}]" for s in all_slots])

    await message.answer(f"📋 Найдены следующие слоты:\n\n{slots_text}\n\nПодтвердить? (yes/no)")
    await state.set_state(AddShiftState.confirming)


@dp.message(StateFilter(AddShiftState.confirming), F.text.lower() == "yes")
async def confirm_yes(message: Message, state: FSMContext):
    data = await state.get_data()
    slots = data.get("slots", [])
    user_id = data.get("user_id")

    if not user_id:
        await message.answer("❌ Ошибка: пользователь не авторизован")
        await state.clear()
        return

    for s in slots:
        await db_add_shift(user_id, s["date"], s["time"], s["status"])

    await message.answer(f"✅ {len(slots)} смен сохранено в БД!")
    await state.clear()


@dp.message(StateFilter(AddShiftState.confirming), F.text.lower() == "no")
async def confirm_no(message: Message, state: FSMContext):
    await message.answer("❌ Отменено")
    await state.clear()


# === MAIN ===
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
