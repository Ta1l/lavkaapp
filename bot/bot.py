# bot/bot.py
import logging
import asyncio
import aiohttp
import os
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from ocr_module import SlotParser  # импорт OCR

# === Конфиг ===
API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"   # <-- замени на токен бота
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
    except Exception as e:
        logger.error(f"Ошибка при запросе токена: {e}")
    return None


# === OCR интеграция ===
def extract_slots_from_user_dir(user_id: int, base_upload_dir: str):
    """
    Берём папку пользователя (сохранённые скрины), запускаем SlotParser,
    возвращаем список слотов.
    """
    user_dir = os.path.join(base_upload_dir, str(user_id))
    parser = SlotParser(base_path=user_dir)
    slots = parser.process_all_screenshots()
    result = []
    for s in slots:
        result.append({
            "date": s["date"],
            "start": s["startTime"],
            "end": s["endTime"],
            "status": "Неизвестно"   # статус пока не распознаём
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
        await message.answer("✅ Вход успешен! Используйте /add чтобы загрузить скриншоты.")
    else:
        await message.answer("❌ Ошибка входа. Попробуйте снова: /start")
        await state.clear()


# === Добавление скринов ===
@dp.message(Command("add"))
async def start_adding(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    await state.set_state(SlotState.waiting_for_screens)
    await message.answer("📷 Отправьте скриншоты. Когда закончите — введите /stop")


@dp.message(StateFilter(SlotState.waiting_for_screens), F.photo)
async def handle_screenshot(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    photo = message.photo[-1]
    file_path = os.path.join(user_dir, f"{photo.file_unique_id}.jpg")
    await photo.download(destination_file=file_path)

    await message.answer("✅ Скрин сохранён.")


@dp.message(Command("stop"), StateFilter(SlotState.waiting_for_screens))
async def stop_and_process(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))

    if not os.path.exists(user_dir):
        await message.answer("⚠️ Сначала загрузите скриншоты через /add")
        return

    slots = extract_slots_from_user_dir(user_id, UPLOAD_DIR)
    if not slots:
        await message.answer("❌ Слоты не распознаны. Попробуйте снова.")
        await state.clear()
        return

    await state.update_data(slots=slots)

    text = "📊 Найдены следующие слоты:\n"
    for s in slots:
        text += f"- {s['date']} {s['start']}–{s['end']} ({s['status']})\n"

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_yes"),
             InlineKeyboardButton(text="❌ Отменить", callback_data="confirm_no")]
        ]
    )

    await message.answer(text, reply_markup=kb)
    await state.set_state(SlotState.confirm_slots)


@dp.callback_query(StateFilter(SlotState.confirm_slots), F.data == "confirm_yes")
async def confirm_yes(call: CallbackQuery, state: FSMContext):
    await call.message.answer("📌 Пока загрузка в веб-приложение отключена (этап разработки).")
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
