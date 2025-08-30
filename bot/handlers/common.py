# bot/handlers/common.py
import os
import logging
from typing import List, Dict, Optional, Any

import aiohttp
from aiogram.fsm.state import State, StatesGroup

# импортируем константы из корня пакета bot
from .. import UPLOAD_DIR, WEBAPP_URL

logger = logging.getLogger("lavka.handlers.common")

# --- FSM состояния (используются во всех модулях) ---
class AuthState(StatesGroup):
    waiting_for_login = State()
    waiting_for_password = State()

class SlotState(StatesGroup):
    waiting_for_screens = State()
    confirm_slots = State()

# Простое in-memory хранилище api-ключей (подходит для single-process dev)
# Если нужно — можно заменить на Redis/базу.
USER_API_KEYS: Dict[int, str] = {}

# --- Утилиты ---
def get_user_dir(user_id: int) -> str:
    """
    Возвращает (и создаёт при необходимости) папку для загрузки скринов данного пользователя.
    """
    p = os.path.join(UPLOAD_DIR, str(user_id))
    os.makedirs(p, exist_ok=True)
    return p

def format_slots_text(slots: List[Dict[str, Any]]) -> str:
    """
    Форматирует список слотов в читаемый текст для сообщения пользователю.
    Поддерживает несколько возможных форматов ключей (start/startTime/start_time).
    """
    if not slots:
        return "Слоты не найдены."
    lines = []
    for i, s in enumerate(slots, start=1):
        date = s.get("date")
        start = s.get("startTime") or s.get("start_time") or s.get("start")
        end = s.get("endTime") or s.get("end_time") or s.get("end")
        lines.append(f"{i}. {date} {start}-{end}")
    return "Найдены слоты:\n" + "\n".join(lines)

# --- API helpers (взаимодействие с веб-приложением) ---
async def get_api_key(username: str, password: str) -> Optional[str]:
    """
    Получаем apiKey из веб-приложения. Возвращаем строку либо None.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/auth/get-token",
                json={"username": username, "password": password},
                timeout=10
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # в разных версиях ключ может называться apiKey или api_key
                    return data.get("apiKey") or data.get("api_key")
                else:
                    text = await resp.text()
                    logger.warning("get_api_key failed: %s %s", resp.status, text[:200])
                    return None
    except Exception as e:
        logger.exception("Ошибка при запросе apiKey: %s", e)
        return None

async def add_shift(api_key: str, date: str, start: str, end: str) -> bool:
    """
    Отправка одного слота в веб-приложение.
    NOTE: по умолчанию функция готова для работы; загрузка может быть отключена (см. ENV ENABLE_UPLOAD).
    """
    ENABLE_UPLOAD = os.getenv("ENABLE_UPLOAD", "0") == "1"
    if not ENABLE_UPLOAD:
        logger.info("UPLOAD disabled by env (ENABLE_UPLOAD!=1). Skipping add_shift for %s %s-%s", date, start, end)
        return False

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"date": date, "startTime": start, "endTime": end, "assignToSelf": True}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{WEBAPP_URL}/api/shifts", headers=headers, json=payload, timeout=10) as resp:
                if resp.status in (200, 201):
                    return True
                else:
                    text = await resp.text()
                    logger.warning("add_shift failed: %s %s", resp.status, text[:200])
                    return False
    except Exception as e:
        logger.exception("Ошибка при add_shift: %s", e)
        return False
