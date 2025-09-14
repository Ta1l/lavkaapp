# bot/handlers/common.py
import logging
from typing import List, Dict, Optional, Any
from io import BytesIO

import aiohttp
from aiogram.fsm.state import State, StatesGroup

from .. import WEBAPP_URL

logger = logging.getLogger("lavka.handlers.common")

# --- FSM состояния ---
class AuthState(StatesGroup):
    waiting_for_login = State()
    waiting_for_password = State()

class SlotState(StatesGroup):
    waiting_for_screens = State()
    confirm_slots = State()

# Хранилище данных пользователей в памяти
class UserStorage:
    def __init__(self):
        self.api_keys: Dict[int, str] = {}
        self.screenshots: Dict[int, List[BytesIO]] = {}
    
    def set_api_key(self, user_id: int, api_key: str):
        self.api_keys[user_id] = api_key
    
    def get_api_key(self, user_id: int) -> Optional[str]:
        return self.api_keys.get(user_id)
    
    def is_authenticated(self, user_id: int) -> bool:
        return user_id in self.api_keys
    
    def add_screenshot(self, user_id: int, screenshot: BytesIO):
        if user_id not in self.screenshots:
            self.screenshots[user_id] = []
        self.screenshots[user_id].append(screenshot)
    
    def get_screenshots(self, user_id: int) -> List[BytesIO]:
        return self.screenshots.get(user_id, [])
    
    def clear_screenshots(self, user_id: int):
        if user_id in self.screenshots:
            # Закрываем все BytesIO объекты
            for bio in self.screenshots[user_id]:
                bio.close()
            del self.screenshots[user_id]
    
    def logout(self, user_id: int):
        if user_id in self.api_keys:
            del self.api_keys[user_id]
        self.clear_screenshots(user_id)

# Глобальное хранилище
user_storage = UserStorage()

# --- Утилиты ---
def format_slots_text(slots: List[Dict[str, Any]]) -> str:
    """
    Форматирует список слотов в читаемый текст для сообщения пользователю.
    """
    if not slots:
        return "Слоты не найдены."
    
    lines = ["📊 Найдены следующие слоты:\n"]
    for i, s in enumerate(slots, start=1):
        date = s.get("date")
        start = s.get("startTime") or s.get("start_time") or s.get("start")
        end = s.get("endTime") or s.get("end_time") or s.get("end")
        lines.append(f"{i}. 📅 {date} ⏰ {start} - {end}")
    
    lines.append(f"\n📌 Всего слотов: {len(slots)}")
    return "\n".join(lines)

# --- API helpers ---
async def get_api_key(username: str, password: str) -> Optional[str]:
    """
    Получаем apiKey из веб-приложения после проверки логина/пароля.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/auth/get-token",
                json={"username": username, "password": password},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    api_key = data.get("apiKey") or data.get("api_key")
                    logger.info(f"Successfully got API key for user {username}")
                    return api_key
                else:
                    text = await resp.text()
                    logger.warning(f"Failed to get API key: {resp.status} - {text[:200]}")
                    return None
    except Exception as e:
        logger.exception(f"Error getting API key: {e}")
        return None

async def add_shift(api_key: str, date: str, start: str, end: str) -> bool:
    """
    Отправка одного слота в веб-приложение.
    """
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
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/shifts",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status in (200, 201):
                    logger.info(f"Successfully added shift: {date} {start}-{end}")
                    return True
                else:
                    text = await resp.text()
                    logger.warning(f"Failed to add shift: {resp.status} - {text[:200]}")
                    return False
    except Exception as e:
        logger.exception(f"Error adding shift: {e}")
        return False