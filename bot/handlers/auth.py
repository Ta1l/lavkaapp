# bot/handlers/auth.py
import logging
from aiogram import types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram import Dispatcher

from .common import AuthState, USER_API_KEYS, get_api_key

logger = logging.getLogger("lavka.handlers.auth")


def register_handlers(dp: Dispatcher):
    """
    Регистрируем обработчики аутентификации.
    """
    dp.message.register(start_cmd, Command("start"))
    dp.message.register(login_input, StateFilter(AuthState.waiting_for_login))
    dp.message.register(password_input, StateFilter(AuthState.waiting_for_password))


# --- Handlers ---
async def start_cmd(message: types.Message, state: FSMContext):
    """
    /start -> просит логин
    """
    await state.set_state(AuthState.waiting_for_login)
    await message.answer("Введите логин (username):")


async def login_input(message: types.Message, state: FSMContext):
    await state.update_data(login=message.text.strip())
    await state.set_state(AuthState.waiting_for_password)
    await message.answer("Введите пароль:")


async def password_input(message: types.Message, state: FSMContext):
    data = await state.get_data()
    login = data.get("login")
    password = message.text.strip()
    if not login:
        await message.answer("Не удалось получить логин. Повторите: /start")
        await state.clear()
        return

    await message.answer("Пробую получить API-ключ...")
    api_key = await get_api_key(login, password)
    if api_key:
        # Сохраняем api_key в простое in-memory хранилище
        USER_API_KEYS[message.from_user.id] = api_key
        await message.answer("✅ Вход успешен! Теперь используйте /add чтобы загрузить скриншоты.")
        logger.info("User %s logged in, api_key received", login)
    else:
        await message.answer("❌ Ошибка входа — проверьте логин/пароль и попробуйте снова (/start).")
        logger.info("Auth failed for user %s", login)

    await state.clear()
