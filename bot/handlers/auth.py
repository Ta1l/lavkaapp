# bot/handlers/auth.py
import logging
from aiogram import types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram import Dispatcher

from .common import AuthState, user_storage, get_api_key

logger = logging.getLogger("lavka.handlers.auth")


def register_handlers(dp: Dispatcher):
    """Регистрируем обработчики аутентификации."""
    dp.message.register(start_cmd, Command("start"))
    dp.message.register(login_input, StateFilter(AuthState.waiting_for_login))
    dp.message.register(password_input, StateFilter(AuthState.waiting_for_password))
    dp.message.register(status_cmd, Command("status"))
    dp.message.register(logout_cmd, Command("logout"))
    dp.message.register(help_cmd, Command("help"))


async def start_cmd(message: types.Message, state: FSMContext):
    """Команда /start - начало работы с ботом."""
    user_id = message.from_user.id
    
    # Проверяем, авторизован ли пользователь
    if user_storage.is_authenticated(user_id):
        await message.answer(
            "✅ Вы уже авторизованы!\n\n"
            "Используйте:\n"
            "• /add - для загрузки скриншотов со слотами\n"
            "• /status - проверить статус\n"
            "• /logout - выйти из системы\n"
            "• /help - справка"
        )
        return
    
    await state.set_state(AuthState.waiting_for_login)
    await message.answer(
        "👋 Добро пожаловать в бот для загрузки слотов!\n\n"
        "Для начала работы необходимо войти в систему.\n"
        "Используйте логин и пароль от сайта slotworker.ru\n\n"
        "📝 Введите ваш логин:"
    )


async def login_input(message: types.Message, state: FSMContext):
    """Обработка ввода логина."""
    login = message.text.strip()
    
    if not login:
        await message.answer("❌ Логин не может быть пустым. Попробуйте еще раз:")
        return
    
    await state.update_data(login=login)
    await state.set_state(AuthState.waiting_for_password)
    await message.answer("🔐 Теперь введите пароль:")


async def password_input(message: types.Message, state: FSMContext):
    """Обработка ввода пароля и авторизация."""
    data = await state.get_data()
    login = data.get("login")
    password = message.text.strip()
    
    if not login:
        await message.answer("❌ Произошла ошибка. Начните заново: /start")
        await state.clear()
        return
    
    if not password:
        await message.answer("❌ Пароль не может быть пустым. Попробуйте еще раз:")
        return

    # Показываем процесс проверки
    msg = await message.answer("🔄 Проверяю данные...")
    
    # Получаем API ключ
    api_key = await get_api_key(login, password)
    
    if api_key:
        # Сохраняем API ключ
        user_storage.set_api_key(message.from_user.id, api_key)
        
        await msg.edit_text(
            "✅ Вход выполнен успешно!\n\n"
            "Теперь вы можете:\n"
            "• Использовать /add для загрузки скриншотов со слотами\n"
            "• Использовать /status для проверки статуса\n"
            "• Использовать /help для получения справки"
        )
        logger.info(f"User {login} (ID: {message.from_user.id}) logged in successfully")
    else:
        await msg.edit_text(
            "❌ Ошибка входа!\n\n"
            "Проверьте правильность логина и пароля.\n"
            "Убедитесь, что вы зарегистрированы на сайте slotworker.ru\n\n"
            "Попробуйте снова: /start"
        )
        logger.warning(f"Failed login attempt for user {login}")

    await state.clear()


async def status_cmd(message: types.Message):
    """Команда /status - проверка статуса авторизации."""
    user_id = message.from_user.id
    
    if user_storage.is_authenticated(user_id):
        screenshots_count = len(user_storage.get_screenshots(user_id))
        status_text = "✅ Вы авторизованы в системе\n\n"
        
        if screenshots_count > 0:
            status_text += f"📷 Загружено скриншотов: {screenshots_count}\n"
            status_text += "Используйте /stop для обработки"
        else:
            status_text += "Используйте /add для загрузки слотов"
        
        await message.answer(status_text)
    else:
        await message.answer(
            "❌ Вы не авторизованы\n\n"
            "Используйте /start для входа в систему"
        )


async def logout_cmd(message: types.Message, state: FSMContext):
    """Команда /logout - выход из системы."""
    user_id = message.from_user.id
    
    if user_storage.is_authenticated(user_id):
        user_storage.logout(user_id)
        await message.answer(
            "👋 Вы вышли из системы\n\n"
            "Для повторного входа используйте /start"
        )
        logger.info(f"User {user_id} logged out")
    else:
        await message.answer("Вы не были авторизованы")
    
    await state.clear()


async def help_cmd(message: types.Message):
    """Команда /help - справка по боту."""
    help_text = """
📖 **Справка по использованию бота**

**Основные команды:**
• /start - начать работу / войти в систему
• /add - загрузить скриншоты со слотами
• /stop - завершить загрузку и обработать скриншоты
• /status - проверить статус авторизации
• /logout - выйти из системы
• /help - показать эту справку

**Как использовать:**
1. Зарегистрируйтесь на сайте slotworker.ru
2. Войдите в бот используя /start
3. Введите логин и пароль от сайта
4. Используйте /add для начала загрузки скриншотов
5. Отправьте скриншоты со слотами
6. Используйте /stop для обработки
7. Подтвердите отправку слотов на сайт

**Важно:**
• Бот автоматически игнорирует отмененные слоты
• Можно загружать несколько скриншотов за раз
• После обработки скриншоты удаляются из памяти
"""
    await message.answer(help_text, parse_mode="Markdown")