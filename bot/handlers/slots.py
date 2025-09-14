# bot/handlers/slots.py
import logging
import asyncio
from io import BytesIO
from typing import List, Dict, Any

from aiogram import types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram import Dispatcher, Bot

from .common import SlotState, user_storage, format_slots_text, add_shift
from ..ocr_module import SlotParser, MemorySlotParser  # Добавляем импорт MemorySlotParser

logger = logging.getLogger("lavka.handlers.slots")


def register_handlers(dp: Dispatcher):
    """Регистрация обработчиков для работы со слотами."""
    dp.message.register(cmd_add, Command("add"))
    dp.message.register(handle_screenshot, StateFilter(SlotState.waiting_for_screens), F.photo)
    dp.message.register(cmd_stop, Command("stop"), StateFilter(SlotState.waiting_for_screens))
    dp.callback_query.register(confirm_yes, StateFilter(SlotState.confirm_slots), F.data == "confirm_yes")
    dp.callback_query.register(confirm_no, StateFilter(SlotState.confirm_slots), F.data == "confirm_no")


async def cmd_add(message: types.Message, state: FSMContext):
    """Команда /add - начало загрузки скриншотов."""
    user_id = message.from_user.id
    
    # Проверяем авторизацию
    if not user_storage.is_authenticated(user_id):
        await message.answer(
            "❌ Необходима авторизация!\n\n"
            "Используйте /start для входа в систему"
        )
        return
    
    # Очищаем старые скриншоты
    user_storage.clear_screenshots(user_id)
    
    await state.set_state(SlotState.waiting_for_screens)
    await message.answer(
                "📷 **Режим загрузки скриншотов активирован**\n\n"
        "Отправьте скриншоты со слотами (можно несколько).\n"
        "После загрузки всех скриншотов используйте /stop\n\n"
        "❗ Отмененные слоты будут автоматически исключены",
        parse_mode="Markdown"
    )
    logger.info(f"User {user_id} started uploading screenshots")


async def handle_screenshot(message: types.Message, state: FSMContext, bot: Bot):
    """Обработка загруженного скриншота."""
    user_id = message.from_user.id
    
    try:
        # Получаем файл
        photo = message.photo[-1]  # Берем самое большое разрешение
        file = await bot.get_file(photo.file_id)
        
        # Загружаем в память
        bio = BytesIO()
        await bot.download_file(file.file_path, bio)
        bio.seek(0)  # Возвращаем указатель в начало
        
        # Сохраняем в памяти для пользователя
        user_storage.add_screenshot(user_id, bio)
        
        count = len(user_storage.get_screenshots(user_id))
        await message.answer(
            f"✅ Скриншот #{count} сохранен\n"
            f"Отправьте еще или используйте /stop для обработки"
        )
        logger.info(f"Screenshot saved for user {user_id}, total: {count}")
        
    except Exception as e:
        logger.exception(f"Error saving screenshot for user {user_id}: {e}")
        await message.answer("❌ Ошибка при сохранении скриншота. Попробуйте еще раз.")


async def cmd_stop(message: types.Message, state: FSMContext):
    """Команда /stop - завершение загрузки и обработка скриншотов."""
    user_id = message.from_user.id
    screenshots = user_storage.get_screenshots(user_id)
    
    if not screenshots:
        await message.answer(
            "⚠️ Нет загруженных скриншотов\n\n"
            "Сначала отправьте скриншоты, затем используйте /stop"
        )
        return
    
    # Показываем процесс обработки
    processing_msg = await message.answer(
        f"🔄 Обрабатываю {len(screenshots)} скриншотов...\n"
        "Это может занять несколько секунд"
    )
    
    try:
        # Создаем временный парсер для обработки из памяти
        parser = MemorySlotParser()
        all_slots = []
        cancelled_count = 0
        
        # Обрабатываем каждый скриншот
        for idx, bio in enumerate(screenshots):
            try:
                # Обновляем сообщение о прогрессе
                if idx > 0 and idx % 2 == 0:
                    await processing_msg.edit_text(
                        f"🔄 Обрабатываю скриншоты...\n"
                        f"Прогресс: {idx + 1}/{len(screenshots)}"
                    )
                
                bio.seek(0)  # Убеждаемся, что читаем с начала
                slots = parser.process_screenshot_from_memory(bio, is_last=(idx == len(screenshots) - 1))
                all_slots.extend(slots)
                cancelled_count += parser.cancelled_count
                
            except Exception as e:
                logger.error(f"Error processing screenshot {idx + 1}: {e}")
        
        # Фильтруем уникальные слоты
        unique_slots = []
        seen = set()
        for slot in all_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key)
                unique_slots.append(slot)
        
        # Сортируем по дате и времени
        unique_slots.sort(key=lambda s: (s["date"], s["startTime"]))
        
        if not unique_slots:
            await processing_msg.edit_text(
                "❌ Не удалось распознать активные слоты\n\n"
                f"Обработано скриншотов: {len(screenshots)}\n"
                f"Отмененных слотов: {cancelled_count}\n\n"
                "Попробуйте сделать более четкие скриншоты"
            )
            user_storage.clear_screenshots(user_id)
            await state.clear()
            return
        
        # Показываем результаты
        result_text = format_slots_text(unique_slots)
        if cancelled_count > 0:
            result_text += f"\n\n⚠️ Пропущено отмененных слотов: {cancelled_count}"
        
        # Кнопки подтверждения
        keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
            [
                types.InlineKeyboardButton(text="✅ Отправить на сайт", callback_data="confirm_yes"),
                types.InlineKeyboardButton(text="❌ Отменить", callback_data="confirm_no")
            ]
        ])
        
        await processing_msg.edit_text(result_text, reply_markup=keyboard)
        
        # Сохраняем слоты в состоянии для последующей отправки
        await state.update_data(slots=unique_slots)
        await state.set_state(SlotState.confirm_slots)
        
    except Exception as e:
        logger.exception(f"Error processing screenshots for user {user_id}: {e}")
        await processing_msg.edit_text(
            "❌ Произошла ошибка при обработке скриншотов\n"
            "Попробуйте еще раз"
        )
        user_storage.clear_screenshots(user_id)
        await state.clear()


async def confirm_yes(call: types.CallbackQuery, state: FSMContext):
    """Подтверждение отправки слотов на сайт."""
    await call.answer()
    user_id = call.from_user.id
    
    # Получаем данные
    data = await state.get_data()
    slots = data.get("slots", [])
    api_key = user_storage.get_api_key(user_id)
    
    if not api_key:
        await call.message.edit_text(
            "❌ Ошибка авторизации\n\n"
            "Используйте /start для повторного входа"
        )
        await state.clear()
        return
    
    if not slots:
        await call.message.edit_text("❌ Нет слотов для отправки")
        await state.clear()
        return
    
    # Начинаем отправку
    await call.message.edit_text(
        "🚀 Отправляю слоты на сайт...\n"
        f"Всего слотов: {len(slots)}"
    )
    
    # Отправляем слоты
    success_count = 0
    failed_count = 0
    
    for i, slot in enumerate(slots):
        try:
            # Отправляем слот
            success = await add_shift(
                api_key,
                slot["date"],
                slot["startTime"],
                slot["endTime"]
            )
            
            if success:
                success_count += 1
            else:
                failed_count += 1
            
            # Обновляем прогресс каждые 3 слота
            if (i + 1) % 3 == 0 or (i + 1) == len(slots):
                progress = (i + 1) / len(slots) * 100
                await call.message.edit_text(
                    f"🚀 Отправка слотов...\n\n"
                    f"Прогресс: {progress:.0f}%\n"
                    f"Отправлено: {i + 1}/{len(slots)}"
                )
            
            # Небольшая задержка между запросами
            await asyncio.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Error sending slot: {e}")
            failed_count += 1
    
    # Итоговое сообщение
    result_message = "✅ **Отправка завершена!**\n\n"
    result_message += f"📊 **Статистика:**\n"
    result_message += f"• Всего слотов: {len(slots)}\n"
    result_message += f"• Успешно отправлено: {success_count}\n"
    
    if failed_count > 0:
        result_message += f"• Ошибок: {failed_count}\n"
    
    result_message += f"\n✨ Слоты добавлены на сайт slotworker.ru"
    
    await call.message.edit_text(result_message, parse_mode="Markdown")
    
    # Очищаем данные
    user_storage.clear_screenshots(user_id)
    await state.clear()
    
    logger.info(f"User {user_id} uploaded {success_count}/{len(slots)} slots")


async def confirm_no(call: types.CallbackQuery, state: FSMContext):
    """Отмена отправки слотов."""
    await call.answer()
    user_id = call.from_user.id
    
    await call.message.edit_text(
        "❌ Отправка отменена\n\n"
        "Скриншоты удалены из памяти.\n"
        "Используйте /add для новой попытки"
    )
    
    # Очищаем данные
    user_storage.clear_screenshots(user_id)
    await state.clear()
    
    logger.info(f"User {user_id} cancelled slot upload")