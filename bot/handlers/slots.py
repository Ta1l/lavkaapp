# bot/handlers/slots.py
import logging
import os
from typing import List, Dict, Any

from aiogram import types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram import Dispatcher, Bot

from .common import SlotState, get_user_dir, format_slots_text, USER_API_KEYS, add_shift
# импортируем класс парсера OCR — предполагается, что модуль ocr_module.py присутствует в корне проекта
from ocr_module import SlotParser

logger = logging.getLogger("lavka.handlers.slots")


def register_handlers(dp: Dispatcher):
    dp.message.register(cmd_add, Command("add"))
    # принимаем фото только когда находимся в состоянии ожидания скринов
    dp.message.register(handle_screenshot, StateFilter(SlotState.waiting_for_screens), F.photo)
    dp.message.register(cmd_stop, Command("stop"), StateFilter(SlotState.waiting_for_screens))
    dp.callback_query.register(confirm_yes, StateFilter(SlotState.confirm_slots), F.data == "confirm_yes")
    dp.callback_query.register(confirm_no, StateFilter(SlotState.confirm_slots), F.data == "confirm_no")


# --- Handlers ---
async def cmd_add(message: types.Message, state: FSMContext):
    """
    /add — переводит бота в режим приёма скриншотов.
    """
    user_dir = get_user_dir(message.from_user.id)
    await state.set_state(SlotState.waiting_for_screens)
    await message.answer(
        "Режим приёма скриншотов включён.\n"
        "Отправляйте скриншоты (можно несколько). Когда закончите, введите /stop"
    )
    logger.debug("User %s starts uploading screenshots into %s", message.from_user.id, user_dir)


async def handle_screenshot(message: types.Message, state: FSMContext):
    """
    Сохраняем фото у пользователя. Имя — file_unique_id.jpg
    """
    user_id = message.from_user.id
    user_dir = get_user_dir(user_id)

    # Берём самый большой размер (последний в списке photo)
    photo = message.photo[-1]
    filename = f"{photo.file_unique_id}.jpg"
    file_path = os.path.join(user_dir, filename)

    try:
        await photo.download(destination_file=file_path)
        await message.answer("✅ Скрин сохранён.")
        logger.info("Saved screenshot for user %s -> %s", user_id, file_path)
    except Exception as e:
        logger.exception("Ошибка сохранения фото: %s", e)
        await message.answer("❌ Ошибка при сохранении скрина.")


async def cmd_stop(message: types.Message, state: FSMContext):
    """
    /stop — останавливаем приём, запускаем OCR по папке пользователя,
    показываем результаты и предлагаем подтвердить.
    """
    user_id = message.from_user.id
    user_dir = get_user_dir(user_id)

    # Проверка наличия файлов
    files = [f for f in os.listdir(user_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        await message.answer("⚠️ Не найдено загруженных скринов. Отправьте скриншот командой /add.")
        await state.clear()
        return

    await message.answer("🔎 Начинаю обработку скринов... Ждите пару секунд.")
    try:
        parser = SlotParser(base_path=user_dir)
        raw_slots = parser.process_all_screenshots()
    except Exception as e:
        logger.exception("OCR error for user %s: %s", user_id, e)
        await message.answer("❌ Ошибка распознавания скринов.")
        await state.clear()
        return

    # Нормализуем формат слотов (поддерживаем несколько ключей)
    slots: List[Dict[str, Any]] = []
    for s in raw_slots:
        # Поддержка разных схем ключей: prefer startTime/endTime, но можно и start_time/start/end
        date = s.get("date")
        start = s.get("startTime") or s.get("start_time") or s.get("start")
        end = s.get("endTime") or s.get("end_time") or s.get("end")
        if date and start and end:
            slots.append({
                "date": date,
                "startTime": start,
                "endTime": end,
                "assignToSelf": True
            })

    if not slots:
        await message.answer("❌ Не удалось распознать слоты на присланных скриншотах.")
        await state.clear()
        return

    text = format_slots_text(slots)
    # Кнопки подтверждения
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_yes"),
         types.InlineKeyboardButton(text="❌ Отменить", callback_data="confirm_no")]
    ])

    # Сохраняем найденные слоты в state
    await state.update_data(slots=slots)
    await message.answer(text, reply_markup=keyboard)
    await state.set_state(SlotState.confirm_slots)


async def confirm_yes(call: types.CallbackQuery, state: FSMContext):
    """
    Пользователь подтвердил — здесь можно выгружать в веб. По умолчанию выгрузка отключена.
    Если нужно включить: задайте переменную окружения ENABLE_UPLOAD=1 на сервере.
    """
    await call.answer()  # убираем "часики" у клиента
    data = await state.get_data()
    slots = data.get("slots", [])
    user_id = call.from_user.id
    api_key = USER_API_KEYS.get(user_id)

    if not slots:
        await call.message.answer("Нет слотов для загрузки.")
        await state.clear()
        return

    if not api_key:
        # Пользователь не залогинен — поясняем
        await call.message.answer("⚠️ Вы не вошли в систему (нет api_key). Сначала выполните /start и войдите.")
        await state.clear()
        return

    # Проверяем ENV — если включено, выполняем загрузку; иначе сообщаем, что отключено
    import os
    if os.getenv("ENABLE_UPLOAD", "0") != "1":
        await call.message.answer(
            "ℹ️ Загрузка в веб-приложение сейчас отключена (режим разработки).\n"
            "Если нужно включить выгрузку — поставьте переменную окружения ENABLE_UPLOAD=1 и перезапустите бота."
        )
        await state.clear()
        return

    # Если дошли сюда — выгружаем
    ok = 0
    total = len(slots)
    for s in slots:
        success = await add_shift(api_key, s["date"], s["startTime"], s["endTime"])
        if success:
            ok += 1

    await call.message.answer(f"Завершено. Загружено {ok}/{total} слотов.")
    await state.clear()


async def confirm_no(call: types.CallbackQuery, state: FSMContext):
    await call.answer()
    await call.message.answer("Отменено. Можно начать заново: /add")
    await state.clear()
