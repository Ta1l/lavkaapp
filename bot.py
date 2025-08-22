import logging
import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

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

# --- FSM (состояния) ---
class AuthState(StatesGroup):
    waiting_for_login = State()
    waiting_for_password = State()

class AddShiftState(StatesGroup):
    waiting_for_photos = State()

# --- DB ---
async def db_get_user(login: str, password: str):
    conn = await asyncpg.connect(**DB_CONFIG)
    row = await conn.fetchrow(
        "SELECT id FROM users WHERE username=$1 AND password=$2",
        login, password
    )
    await conn.close()
    return row["id"] if row else None

# --- START ---
@dp.message(Command("start"))
async def start(message: Message, state: FSMContext):
    await message.answer("Введите логин:")
    await state.set_state(AuthState.waiting_for_login)

# --- LOGIN ---
@dp.message(StateFilter(AuthState.waiting_for_login))
async def get_login(message: Message, state: FSMContext):
    await state.update_data(login=message.text.strip())
    await message.answer("Введите пароль:")
    await state.set_state(AuthState.waiting_for_password)

# --- PASSWORD ---
@dp.message(StateFilter(AuthState.waiting_for_password))
async def get_password(message: Message, state: FSMContext):
    data = await state.get_data()
    login = data.get("login")
    password = message.text.strip()

    user_id = await db_get_user(login, password)
    if user_id:
        await message.answer(f"✅ Успешный вход! Добро пожаловать, {login}")
        await state.clear()
    else:
        await message.answer("❌ Неверный логин или пароль. Попробуйте снова: /start")
        await state.clear()

# --- ADD SHIFTS ---
@dp.message(Command("add"))
async def add_shifts(message: Message, state: FSMContext):
    await message.answer("Отправьте скриншоты смен. Когда закончите, напишите /done")
    await state.set_state(AddShiftState.waiting_for_photos)

@dp.message(StateFilter(AddShiftState.waiting_for_photos), F.photo)
async def handle_photos(message: Message, state: FSMContext):
    file_id = message.photo[-1].file_id
    data = await state.get_data()
    photos = data.get("photos", [])
    photos.append(file_id)
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

    # Тут будет OCR и парсинг (пока заглушка)
    await message.answer(f"📸 Получено {len(photos)} скриншотов.\nПодтвердите отправку смен? (yes/no)")

    await state.update_data(confirm=True)

@dp.message(StateFilter(AddShiftState.waiting_for_photos), F.text.lower() == "yes")
async def confirm_yes(message: Message, state: FSMContext):
    data = await state.get_data()
    photos = data.get("photos", [])
    # Заглушка — тут ты добавишь отправку слотов в БД
    await message.answer(f"✅ Смены ({len(photos)} шт.) сохранены на сервере!")
    await state.clear()

@dp.message(StateFilter(AddShiftState.waiting_for_photos), F.text.lower() == "no")
async def confirm_no(message: Message, state: FSMContext):
    await message.answer("❌ Отменено")
    await state.clear()

# --- MAIN ---
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
