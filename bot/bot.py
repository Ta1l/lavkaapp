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

from ocr_module import SlotParser  # OCR парсер

# === Конфиг ===
API_TOKEN = "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM"
WEBAPP_URL = "https://slotworker.ru"   # ⚠️ без лишнего "/"
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
    """Получение API-ключа по логину/паролю"""
    try:
        url = f"{WEBAPP_URL}/api/auth/get-token"
        logger.info(f"🔑 Запрос токена для {username} на {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={"username": username, "password": password},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                text = await response.text()
                logger.info(f"📥 Ответ сервера: status={response.status}, body={text[:200]}...")
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        api_key = data.get("apiKey")
                        if api_key:
                            logger.info(f"✅ Получен apiKey для {username}: {api_key[:10]}...")
                            return api_key
                        else:
                            logger.error(f"❌ В ответе нет apiKey: {data}")
                            return None
                    except Exception as e:
                        logger.error(f"❌ Ошибка парсинга JSON: {e}, text={text}")
                        return None
                else:
                    logger.warning(f"⚠️ Ошибка API аутентификации {response.status}: {text}")
                    return None
    except asyncio.TimeoutError:
        logger.error(f"❌ Таймаут при запросе токена")
    except Exception as e:
        logger.error(f"❌ Ошибка при запросе токена: {type(e).__name__}: {e}")
    return None


async def add_shift(api_key: str, date: str, start: str, end: str) -> bool:
    """Отправка слота в веб-приложение"""
    try:
        url = f"{WEBAPP_URL}/api/shifts"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "date": date,
            "startTime": start,
            "endTime": end,
            "assignToSelf": True
        }
        
        logger.info(f"📤 Отправка слота: {url}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Payload: {payload}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                text = await response.text()
                logger.info(f"📥 Ответ: status={response.status}, body={text[:200]}...")
                
                if response.status in (200, 201):
                    logger.info(f"✅ Слот {date} {start}-{end} добавлен")
                    return True
                else:
                    logger.warning(f"⚠️ Ошибка API добавления слота {response.status}: {text}")
                    return False
    except Exception as e:
        logger.error(f"❌ Ошибка при добавлении слота: {type(e).__name__}: {e}")
    return False


# === OCR интеграция ===
def extract_slots_from_user_dir(user_id: int, upload_dir: str):
    """Запуск OCR для распознавания скриншотов"""
    user_dir = os.path.join(upload_dir, str(user_id))
    parser = SlotParser(base_path=user_dir)
    slots = parser.process_all_screenshots()

    result = []
    for s in slots:
        result.append({
            "date": s["date"],
            "start": s["start_time"],
            "end": s["end_time"],
            "status": s.get("status", "Неизвестно")
        })
    return result


# === Хэндлеры ===
@dp.message(Command("start"))
async def start_cmd(message: Message, state: FSMContext):
    await message.answer("👋 Добро пожаловать! Введите логин:")
    await state.set_state(AuthState.waiting_for_login)


@dp.message(StateFilter(AuthState.waiting_for_login))
async def login_input(message: Message, state: FSMContext):
    await state.update_data(login=message.text.strip())
    await message.answer("🔐 Введите пароль:")
    await state.set_state(AuthState.waiting_for_password)


@dp.message(StateFilter(AuthState.waiting_for_password))
async def password_input(message: Message, state: FSMContext):
    data = await state.get_data()
    login = data.get("login")
    password = message.text.strip()

    await message.answer("🔄 Проверяю данные...")
    
    api_key = await get_api_key(login, password)
    if api_key:
        await state.update_data(api_key=api_key)
        # Проверяем, что ключ сохранился
        check_data = await state.get_data()
        logger.info(f"📝 Сохранен api_key в состоянии: {check_data.get('api_key', 'НЕТ')[:10]}...")
        await message.answer("✅ Вход успешен! Используйте /add чтобы загрузить скриншоты.")
        await state.set_state(None)  # Сбрасываем состояние авторизации
    else:
        await message.answer("❌ Ошибка входа. Проверьте логин/пароль. Попробуйте снова: /start")
        await state.clear()


@dp.message(Command("add"))
async def start_adding(message: Message, state: FSMContext):
    # Проверяем авторизацию
    data = await state.get_data()
    if not data.get("api_key"):
        await message.answer("❌ Сначала войдите в систему: /start")
        return
    
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
    
    try:
        file = await bot.get_file(photo.file_id)
        await bot.download_file(file.file_path, file_path)
        await message.answer("✅ Скрин сохранён.")
        logger.info(f"📸 Сохранен скриншот: {file_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения скриншота: {e}")
        await message.answer("❌ Ошибка при сохранении скриншота.")


@dp.message(Command("stop"), StateFilter(SlotState.waiting_for_screens))
async def stop_and_process(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))

    if not os.path.exists(user_dir) or not os.listdir(user_dir):
        await message.answer("⚠️ Сначала загрузите скриншоты через /add")
        await state.set_state(None)
        return

    await message.answer("🔄 Обрабатываю скриншоты...")
    
    try:
        slots = extract_slots_from_user_dir(user_id, UPLOAD_DIR)
        if not slots:
            await message.answer("❌ Слоты не распознаны. Попробуйте снова.")
            await state.set_state(None)
            return

        await state.update_data(slots=slots)

        text = "📊 Найдены следующие слоты:\n\n"
        for i, s in enumerate(slots, 1):
            text += f"{i}. {s['date']} {s['start']}–{s['end']} ({s['status']})\n"

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_yes"),
                 InlineKeyboardButton(text="❌ Отменить", callback_data="confirm_no")]
            ]
        )

        await message.answer(text, reply_markup=kb)
        await state.set_state(SlotState.confirm_slots)
    except Exception as e:
        logger.error(f"❌ Ошибка обработки скриншотов: {e}")
        await message.answer("❌ Ошибка при обработке скриншотов.")
        await state.set_state(None)


@dp.callback_query(StateFilter(SlotState.confirm_slots), F.data == "confirm_yes")
async def confirm_yes(call: CallbackQuery, state: FSMContext):
    await call.answer()
    
    data = await state.get_data()
    api_key = data.get("api_key")
    slots = data.get("slots", [])
    
    if not api_key:
        await call.message.answer("❌ Ошибка: нет API-ключа. Пожалуйста, войдите заново: /start")
        await state.clear()
        return
    
    logger.info(f"📤 Начинаю отправку {len(slots)} слотов с api_key: {api_key[:10]}...")
    
    await call.message.edit_text("🔄 Отправляю слоты...")
    
    ok_count = 0
    for s in slots:
        if await add_shift(api_key, s["date"], s["start"], s["end"]):
            ok_count += 1
        await asyncio.sleep(0.5)  # Небольшая задержка между запросами

    await call.message.answer(f"✅ Загружено {ok_count} из {len(slots)} слотов в расписание.")
    
    # Очищаем папку со скриншотами
    user_id = call.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    try:
        for file in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, file))
        logger.info(f"🗑️ Очищена папка {user_dir}")
    except Exception as e:
        logger.error(f"❌ Ошибка очистки папки: {e}")
    
    await state.set_state(None)  # Сохраняем api_key, но сбрасываем состояние


@dp.callback_query(StateFilter(SlotState.confirm_slots), F.data == "confirm_no")
async def confirm_no(call: CallbackQuery, state: FSMContext):
    await call.answer()
    await call.message.edit_text("❌ Отменено. Начните заново: /add")
    
    # Очищаем папку со скриншотами
    user_id = call.from_user.id
    user_dir = os.path.join(UPLOAD_DIR, str(user_id))
    try:
        for file in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, file))
        logger.info(f"🗑️ Очищена папка {user_dir}")
    except Exception as e:
        logger.error(f"❌ Ошибка очистки папки: {e}")
    
    await state.set_state(None)  # Сохраняем api_key, но сбрасываем состояние


@dp.message(Command("status"))
async def check_status(message: Message, state: FSMContext):
    data = await state.get_data()
    api_key = data.get("api_key")
    
    if api_key:
        await message.answer(f"✅ Вы авторизованы. API-ключ: {api_key[:10]}...")
    else:
        await message.answer("❌ Вы не авторизованы. Используйте /start для входа.")


@dp.message(Command("logout"))
async def logout(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("👋 Вы вышли из системы. Для входа используйте /start")


# === MAIN ===
async def main():
    logger.info("🚀 Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())