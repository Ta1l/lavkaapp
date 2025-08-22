import asyncio
import json
import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from ocr_parser import parse_slots

API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Подключение к БД
async def get_db_pool():
    return await asyncpg.create_pool(
    user="lavka_user",
    password="hw6uxxs9*Hz5", # Пароль, который мы устанавливали
    database="schedule_db",
    host="localhost"
)

# Хранилище логинов (для примера)
user_sessions = {}

# Стартовая команда
@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Введите логин:")

# Логин
@dp.message(F.text & (lambda m: m.chat.id not in user_sessions))
async def login_step(message: Message):
    # Сохраняем логин и просим пароль
    user_sessions[message.chat.id] = {"login": message.text}
    await message.answer("Введите пароль:")

# Пароль
@dp.message(F.text & (lambda m: "login" in user_sessions.get(m.chat.id, {}) and "user_id" not in user_sessions[m.chat.id]))
async def password_step(message: Message):
    login = user_sessions[message.chat.id]["login"]
    password = message.text

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, password FROM users WHERE login=$1", login)

    if row and password == row["password"]:  # здесь лучше использовать bcrypt.checkpw
        user_sessions[message.chat.id]["user_id"] = row["id"]
        await message.answer("✅ Успешный вход! Теперь используйте /add для загрузки смен.")
    else:
        await message.answer("❌ Неверный логин или пароль")

# Команда /add
@dp.message(Command("add"))
async def add_slots(message: Message):
    if "user_id" not in user_sessions.get(message.chat.id, {}):
        await message.answer("Сначала войдите через /start")
        return
    await message.answer("Отправьте скриншот со сменами")

# Приём фото
@dp.message(F.photo)
async def handle_photo(message: Message):
    if "user_id" not in user_sessions.get(message.chat.id, {}):
        await message.answer("Сначала войдите через /start")
        return

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = f"/tmp/{file.file_id}.jpg"
    await bot.download_file(file.file_path, destination=file_path)

    slots = parse_slots(file_path)
    if not slots:
        await message.answer("Не удалось распознать слоты")
        return

    # Сохраняем распознанные слоты во временное хранилище
    user_sessions[message.chat.id]["pending_slots"] = slots

    result = json.dumps(slots, ensure_ascii=False, indent=2)
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Верно", callback_data="confirm_slots")
    kb.button(text="❌ Отмена", callback_data="cancel_slots")
    await message.answer(f"Распознаны слоты:\n```json\n{result}\n```", parse_mode="Markdown", reply_markup=kb.as_markup())

# Подтверждение
@dp.callback_query(F.data == "confirm_slots")
async def confirm_slots(call: CallbackQuery):
    user_id = user_sessions[call.message.chat.id]["user_id"]
    slots = user_sessions[call.message.chat.id].get("pending_slots", [])

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        for slot in slots:
            await conn.execute(
                "INSERT INTO shifts (user_id, start_time, end_time, status, address) VALUES ($1, $2, $3, $4, $5)",
                user_id, slot["start_time"], slot["end_time"], slot["status"], slot.get("address", "")
            )

    await call.message.answer("✅ Смены успешно добавлены в систему")
    user_sessions[call.message.chat.id].pop("pending_slots", None)

# Отмена
@dp.callback_query(F.data == "cancel_slots")
async def cancel_slots(call: CallbackQuery):
    user_sessions[call.message.chat.id].pop("pending_slots", None)
    await call.message.answer("❌ Смена отменена")


async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
