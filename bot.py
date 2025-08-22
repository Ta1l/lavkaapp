import asyncio
import bcrypt
import logging
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
import asyncpg

# ----------------- НАСТРОЙКИ -----------------
API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"

DB_CONFIG = {
    "user": "lavka_user",         # см. src/lib/db.ts
    "password": "hw6uxxs9*Hz5",
    "database": "schedule_db",
    "host": "localhost",          # заменить на адрес сервера
    "port": 5432
}
# ---------------------------------------------

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# FSM для входа
class LoginForm(StatesGroup):
    username = State()
    password = State()

# Подключение к БД
async def get_db_pool():
    return await asyncpg.create_pool(**DB_CONFIG)

# ----------------- ХЕНДЛЕРЫ -----------------
@dp.message(F.text == "/start")
async def start_cmd(message: Message, state: FSMContext):
    await message.answer("Привет! Введите ваш логин:")
    await state.set_state(LoginForm.username)

@dp.message(LoginForm.username)
async def process_username(message: Message, state: FSMContext):
    await state.update_data(username=message.text.strip())
    await message.answer("Теперь введите пароль:")
    await state.set_state(LoginForm.password)

@dp.message(LoginForm.password)
async def process_password(message: Message, state: FSMContext):
    user_data = await state.get_data()
    username = user_data["username"]
    password = message.text.strip()

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, username, password FROM users WHERE username = $1", username)

    if not row:
        await message.answer("❌ Пользователь не найден.")
        await state.clear()
        return

    # проверяем хэш через bcrypt
    if bcrypt.checkpw(password.encode(), row["password"].encode()):
        await message.answer(f"✅ Успешный вход! Добро пожаловать, {row['username']}.")
    else:
        await message.answer("❌ Неверный пароль.")

    await state.clear()

# ----------------- MAIN -----------------
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
