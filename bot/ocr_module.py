# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для нового формата скриншотов.
Полная совместимость со старым API.
"""

import os
import re
import logging
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytesseract

logger = logging.getLogger("lavka.ocr_module")
logger.setLevel(logging.INFO)

import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

# --------- Константы ---------
FIXED_YEAR = 2025
FIXED_MONTH = 10

# Расширенные паттерны для поиска времени
TIME_PATTERNS = [
    # Строгий паттерн HH:MM - HH:MM с разными тире
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    # С точками HH.MM - HH.MM
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    # С пробелами HH MM - HH MM
    re.compile(r"(\d{1,2})\s+(\d{2})\s*[-–—]\s*(\d{1,2})\s+(\d{2})"),
    # Слитно HHMM-HHMM
    re.compile(r"(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})"),
    # Без разделителей между часами и минутами
    re.compile(r"(\d{1,2})(\d{2})\s*[-–—]\s*(\d{1,2})(\d{2})"),
]

# Паттерн для поиска одиночного времени
SINGLE_TIME_RE = re.compile(r"(\d{1,2})[:\.](\d{2})")

DIGIT_RE = re.compile(r"\d{1,2}")


# --------- Вспомогательные функции ---------
def _read_image(image_input: Any) -> Optional[np.ndarray]:
    """Принимает путь str, BytesIO, bytes, PIL.Image -> возвращает BGR numpy array."""
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.error(f"Image file not found: {image_input}")
                return None
            img = cv2.imread(image_input)
            if img is None:
                logger.error(f"Failed to read image with cv2: {image_input}")
            return img
        if isinstance(image_input, bytes):
            bio = BytesIO(image_input)
            pil = Image.open(bio).convert("RGB")
            arr = np.array(pil)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if isinstance(image_input, BytesIO):
            image_input.seek(0)
            pil = Image.open(image_input).convert("RGB")
            arr = np.array(pil)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        logger.error(f"Unknown image input type: {type(image_input)}")
    except Exception as e:
        logger.error(f"_read_image error: {e}", exc_info=True)
    return None


def _preprocess_for_ocr(bgr: np.ndarray, scale: float = 1.4) -> np.ndarray:
    """Стандартная предобработка для OCR."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    new_w = max(100, int(w * scale))
    new_h = max(60, int(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)
    return bw


def _normalize_text_for_time(text: str) -> str:
    """Нормализует текст для поиска времени."""
    if not text:
        return ""
    
    # Заменяем похожие символы на цифры
    replacements = {
        "O": "0", "o": "0", "Q": "0", "О": "0", "о": "0",
        "l": "1", "I": "1", "|": "1", "í": "1", "i": "1",
        "S": "5", "s": "5", "З": "3", "з": "3",
        "B": "8", "В": "8", "в": "8",
        "б": "6", "Б": "6", "G": "6", "g": "9",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Нормализуем разделители
    text = text.replace("：", ":").replace("∶", ":").replace("․", ":")
    text = text.replace("·", ":").replace("•", ":").replace(",", ":")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    
    return text


def _extract_time_ranges_robust(text: str) -> List[Tuple[str, str]]:
    """Робастное извлечение временных диапазонов."""
    if not text:
        return []
    
    text = _normalize_text_for_time(text)
    ranges = []
    
    # Пробуем все паттерны
    for pattern in TIME_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            try:
                h1, m1, h2, m2 = match
                h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
                
                if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                    start = f"{h1:02d}:{m1:02d}"
                    end = f"{h2:02d}:{m2:02d}"
                    ranges.append((start, end))
                    logger.debug(f"Found time range: {start} - {end}")
            except Exception as e:
                logger.debug(f"Failed to parse time match: {match}, error: {e}")
    
    # Если не нашли диапазоны, ищем отдельные времена
    if not ranges:
        times = []
        matches = SINGLE_TIME_RE.findall(text)
        for h, m in matches:
            try:
                h_int, m_int = int(h), int(m)
                if 0 <= h_int < 24 and 0 <= m_int < 60:
                    times.append(f"{h_int:02d}:{m_int:02d}")
            except:
                pass
        
        # Группируем попарно
        if len(times) >= 2:
            for i in range(0, len(times) - 1, 2):
                ranges.append((times[i], times[i + 1]))
                logger.debug(f"Paired times: {times[i]} - {times[i + 1]}")
    
    # Уникализация
    seen = set()
    unique = []
    for r in ranges:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    
    return unique


# --------- Основной класс парсера ---------
class NewFormatSlotParser:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.cancelled_count = 0  # Для совместимости
        
    def _find_yellow_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Поиск жёлтой области для определения дня."""
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            
            # Более широкий диапазон для жёлтого/оранжевого
            masks = []
            # Жёлтый
            masks.append(cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255])))
            # Оранжевый
            masks.append(cv2.inRange(hsv, np.array([5, 50, 50]), np.array([15, 255, 255])))
            # Светло-жёлтый
            masks.append(cv2.inRange(hsv, np.array([20, 30, 100]), np.array([30, 255, 255])))
            
            mask = cv2.bitwise_or(masks[0], masks[1])
            if len(masks) > 2:
                mask = cv2.bitwise_or(mask, masks[2])
            
            H, W = mask.shape[:2]
            mask[int(H * 0.45):, :] = 0  # Только верхняя часть
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area < 200:
                return None
            
            return cv2.boundingRect(best)
            
        except Exception as e:
            logger.debug(f"_find_yellow_bbox error: {e}")
        return None
    
    def _extract_day_number(self, bgr: np.ndarray) -> Optional[int]:
        """Извлечение номера дня из изображения."""
        # Пробуем найти жёлтый блок
        bbox = self._find_yellow_bbox(bgr)
        if bbox:
            x, y, w, h = bbox
            pad = 10
            roi = bgr[max(0, y-pad):min(bgr.shape[0], y+h+pad), 
                     max(0, x-pad):min(bgr.shape[1], x+w+pad)]
        else:
            # Берём верхнюю треть
            h = bgr.shape[0]
            roi = bgr[:h//3, :]
        
        try:
            # OCR с разными методами
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Метод 1: простой OCR
            text = pytesseract.image_to_string(gray, lang='eng')
            numbers = re.findall(r'\b(\d{1,2})\b', text)
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= 31:
                    return num
            
            # Метод 2: с предобработкой
            processed = _preprocess_for_ocr(roi)
            text = pytesseract.image_to_string(processed, lang='eng')
            numbers = re.findall(r'\b(\d{1,2})\b', text)
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= 31:
                    return num
                    
        except Exception as e:
            logger.debug(f"Day extraction error: {e}")
        
        return None
    
    def process_image(self, image_input: Any) -> List[Dict]:
        """Обработка одного изображения."""
        img = _read_image(image_input)
        if img is None:
            logger.error("Cannot read image")
            return []
        
        # 1. Извлекаем день
        day = self._extract_day_number(img)
        if day is None:
            logger.warning("Day not detected, using default: 1")
            day = 1
        else:
            logger.info(f"Detected day: {day}")
        
        # 2. Формируем дату
        try:
            slot_date = date(FIXED_YEAR, FIXED_MONTH, day)
            iso_date = slot_date.isoformat()
        except:
            logger.error(f"Invalid day: {day}")
            return []
        
        # 3. Ищем временные слоты разными методами
        all_ranges = []
        
        # Метод 1: OCR всего изображения
        try:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            text_full = pytesseract.image_to_string(pil, lang='eng+rus')
            ranges = _extract_time_ranges_robust(text_full)
            all_ranges.extend(ranges)
            if ranges:
                logger.info(f"Method 1 (full): found {len(ranges)} ranges")
        except Exception as e:
            logger.debug(f"Method 1 failed: {e}")
        
        # Метод 2: Нижняя часть (где обычно слоты)
        if not all_ranges:
            try:
                h = img.shape[0]
                bottom = img[h//3:, :]
                pil = Image.fromarray(cv2.cvtColor(bottom, cv2.COLOR_BGR2RGB))
                text_bottom = pytesseract.image_to_string(pil, lang='eng+rus')
                ranges = _extract_time_ranges_robust(text_bottom)
                all_ranges.extend(ranges)
                if ranges:
                    logger.info(f"Method 2 (bottom): found {len(ranges)} ranges")
            except Exception as e:
                logger.debug(f"Method 2 failed: {e}")
        
        # Метод 3: С предобработкой
        if not all_ranges:
            try:
                processed = _preprocess_for_ocr(img)
                pil = Image.fromarray(processed)
                text_proc = pytesseract.image_to_string(pil, lang='eng')
                ranges = _extract_time_ranges_robust(text_proc)
                all_ranges.extend(ranges)
                if ranges:
                    logger.info(f"Method 3 (processed): found {len(ranges)} ranges")
            except Exception as e:
                logger.debug(f"Method 3 failed: {e}")
        
        # Метод 4: image_to_data для точного извлечения
        if not all_ranges:
            try:
                processed = _preprocess_for_ocr(img)
                pil = Image.fromarray(processed)
                data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, lang='eng')
                
                # Собираем весь текст
                text_parts = []
                for i, txt in enumerate(data['text']):
                    if txt and txt.strip():
                        text_parts.append(txt.strip())
                
                full_text = " ".join(text_parts)
                ranges = _extract_time_ranges_robust(full_text)
                all_ranges.extend(ranges)
                if ranges:
                    logger.info(f"Method 4 (data): found {len(ranges)} ranges")
                    
            except Exception as e:
                logger.debug(f"Method 4 failed: {e}")
        
        # Уникализация
        seen = set()
        slots = []
        for start, end in all_ranges:
            if (start, end) not in seen:
                seen.add((start, end))
                slots.append({
                    "date": iso_date,
                    "startTime": start,
                    "endTime": end,
                    "assignToSelf": True
                })
        
        logger.info(f"Total found {len(slots)} unique slot(s) for {iso_date}")
        return slots


# --------- Класс MemorySlotParser для полной совместимости ---------
class MemorySlotParser:
    """Парсер для работы со скриншотами из памяти. Полная совместимость со старым API."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.parser = NewFormatSlotParser(debug=debug)
        
        # Атрибуты для совместимости со старым кодом
        self.base_path = ""
        self.cancelled_count = 0
        self.accumulated_slots = []
        self.screenshots_count = 0
        self.last_error = None
        
        # Атрибуты из старого кода
        self.months = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        self.status_map = {
            "выполнен с опозданием": "Выполнен с опозданием",
            "выполнен": "Выполнен",
            "отменен": "Отменён",
            "отменён": "Отменён",
            "отмен": "Отменён"
        }
        
    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """Обработка одного скриншота из памяти (основной метод для совместимости)."""
        try:
            logger.info(f"Processing screenshot from memory, is_last={is_last}")
            
            # Обрабатываем изображение
            slots = self.parser.process_image(image_bytes)
            
            # Накапливаем слоты
            self.accumulated_slots.extend(slots)
            self.screenshots_count += 1
            
            # Если последний скриншот, возвращаем все накопленные
            if is_last:
                return self.get_all_slots()
            
            return slots
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in process_screenshot_from_memory: {e}", exc_info=True)
            return []
    
    def process_screenshot(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """Алиас для process_screenshot_from_memory."""
        return self.process_screenshot_from_memory(image_bytes, is_last)
    
    def process_image(self, image_input: Any) -> List[Dict]:
        """Обработка одного изображения."""
        try:
            return self.parser.process_image(image_input)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in process_image: {e}")
            return []
    
    def process_image_bytes(self, image_bytes: bytes) -> List[Dict]:
        """Обработка изображения из байтов."""
        try:
            return self.parser.process_image(image_bytes)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error in process_image_bytes: {e}")
            return []
    
    def preprocess_image_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Предобработка изображения (для совместимости)."""
        try:
            return _preprocess_for_ocr(image_array)
        except Exception as e:
            logger.error(f"Error in preprocess_image_array: {e}")
            return None
    
    def get_all_slots(self) -> List[Dict]:
        """Возвращает все накопленные слоты."""
        seen = set()
        unique = []
        for slot in self.accumulated_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key)
                unique.append(slot)
        unique.sort(key=lambda s: (s["date"], s["startTime"]))
        return unique
    
    def clear(self):
        """Очищает накопленные данные."""
        self.accumulated_slots = []
        self.screenshots_count = 0
        self.last_error = None
        self.cancelled_count = 0
    
    def reset(self):
        """Алиас для clear."""
        self.clear()
    
    # Методы из старого кода для полной совместимости
    def parse_time_in_text(self, text: str) -> Optional[Tuple[str, str]]:
        """Поиск времени в тексте (для совместимости)."""
        ranges = _extract_time_ranges_robust(text)
        return ranges[0] if ranges else None
    
    def _extract_lines_from_data(self, data: Dict) -> List[Dict]:
        """Извлечение строк из OCR данных (для совместимости)."""
        try:
            groups = {}
            for i in range(len(data.get("text", []))):
                txt = data["text"][i].strip() if data["text"][i] else ""
                if not txt:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                groups.setdefault(key, []).append(i)
            
            lines = []
            for key, idxs in groups.items():
                idxs.sort(key=lambda j: data["left"][j])
                parts = [data["text"][j].strip() for j in idxs if data["text"][j].strip()]
                if not parts:
                    continue
                text = " ".join(parts)
                top = min(data["top"][j] for j in idxs)
                left = min(data["left"][j] for j in idxs)
                lines.append({"text": text, "y": int(top), "x": int(left)})
            
            lines.sort(key=lambda l: (l["y"], l["x"]))
            return lines
        except Exception as e:
            logger.error(f"Error extracting lines: {e}")
            return []
    
    def process_all_screenshots(self) -> List[Dict]:
        """Для совместимости."""
        return self.get_all_slots()
    
    def get_screenshots_count(self) -> int:
        """Возвращает количество обработанных скриншотов."""
        return self.screenshots_count
    
    def get_last_error(self) -> Optional[str]:
        """Возвращает последнюю ошибку."""
        return self.last_error


# --------- Класс SlotParser для совместимости ---------
class SlotParser:
    """Класс для обработки папки со скриншотами."""
    
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.parser = NewFormatSlotParser(debug=debug)
        self.cancelled_count = 0
        
        # Атрибуты из старого кода
        self.months = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        
    def process_all_screenshots(self) -> List[Dict]:
        """Обрабатывает все скриншоты в папке."""
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}")
            return []
        
        files = [
            os.path.join(self.base_path, f)
            for f in os.listdir(self.base_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not files:
            logger.warning(f"No image files found in: {self.base_path}")
            return []
        
        all_slots = []
        files.sort()
        
        for filepath in files:
            try:
                slots = self.parser.process_image(filepath)
                all_slots.extend(slots)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        # Уникализация
        seen = set()
        unique = []
        for slot in all_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key)
                unique.append(slot)
        
        unique.sort(key=lambda s: (s["date"], s["startTime"]))
        return unique


__all__ = ["SlotParser", "MemorySlotParser", "NewFormatSlotParser"]