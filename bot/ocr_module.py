# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для нового формата скриншотов.
Исправлена проблема с определением дня.
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
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    re.compile(r"(\d{1,2})\s+(\d{2})\s*[-–—]\s*(\d{1,2})\s+(\d{2})"),
    re.compile(r"(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})"),
]

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
    
    replacements = {
        "O": "0", "o": "0", "О": "0", "о": "0",
        "l": "1", "I": "1", "|": "1", "í": "1",
        "S": "5", "s": "5", "З": "3",
        "B": "8", "В": "8", "в": "8",
        "б": "6", "Б": "6", "G": "6",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
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
        
        if len(times) >= 2:
            for i in range(0, len(times) - 1, 2):
                ranges.append((times[i], times[i + 1]))
    
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
        self.cancelled_count = 0
        
    def _find_yellow_region(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Улучшенный поиск жёлтой/оранжевой области."""
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            
            # Комбинируем несколько масок для разных оттенков жёлтого/оранжевого
            masks = []
            
            # Жёлтый диапазон
            masks.append(cv2.inRange(hsv, np.array([20, 50, 50]), np.array([35, 255, 255])))
            
            # Оранжевый диапазон
            masks.append(cv2.inRange(hsv, np.array([10, 50, 50]), np.array([20, 255, 255])))
            
            # Светло-жёлтый
            masks.append(cv2.inRange(hsv, np.array([25, 30, 100]), np.array([35, 255, 255])))
            
            # Тёмно-оранжевый
            masks.append(cv2.inRange(hsv, np.array([5, 100, 100]), np.array([15, 255, 255])))
            
            # Объединяем все маски
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Ограничиваем поиск верхней половиной изображения
            h, w = combined_mask.shape
            combined_mask[h//2:, :] = 0
            
            # Морфологические операции для очистки
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Находим контуры
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Ищем подходящий контур
            best_contour = None
            best_score = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:  # Слишком маленький
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Проверяем соотношение сторон (должно быть примерно квадратное)
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Вычисляем score (предпочитаем большие области ближе к верху)
                score = area - y * 10
                
                if score > best_score:
                    best_score = score
                    best_contour = (x, y, w, h)
            
            return best_contour
            
        except Exception as e:
            logger.debug(f"_find_yellow_region error: {e}")
            return None
    
    def _extract_day_advanced(self, bgr: np.ndarray) -> Optional[int]:
        """Продвинутое извлечение дня с несколькими методами."""
        day_candidates = []
        
        # Метод 1: Поиск в жёлтой области
        yellow_bbox = self._find_yellow_region(bgr)
        if yellow_bbox:
            x, y, w, h = yellow_bbox
            logger.info(f"Found yellow region at ({x},{y}) size ({w}x{h})")
            
            # Расширяем область поиска
            pad = 15
            roi = bgr[max(0, y-pad):min(bgr.shape[0], y+h+pad), 
                     max(0, x-pad):min(bgr.shape[1], x+w+pad)]
            
            # OCR на области
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Пробуем разные методы обработки
            for method in ['direct', 'thresh', 'adaptive']:
                if method == 'direct':
                    processed = gray
                elif method == 'thresh':
                    _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:  # adaptive
                    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
                
                # Увеличиваем для лучшего OCR
                processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # OCR только цифры
                config = "--psm 8 -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(processed, config=config, lang='eng')
                
                numbers = re.findall(r'\d+', text)
                for num_str in numbers:
                    num = int(num_str)
                    if 1 <= num <= 31:
                        day_candidates.append(num)
                        logger.debug(f"Method 1.{method}: found day {num}")
        
        # Метод 2: Поиск по всей верхней части изображения
        top_third = bgr[:bgr.shape[0]//3, :]
        
        # OCR с разными конфигурациями
        configs = [
            "--psm 6",  # Uniform block
            "--psm 11",  # Sparse text
            "--psm 12",  # Sparse text with OSD
        ]
        
        for config in configs:
            try:
                gray = cv2.cvtColor(top_third, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config=config, lang='eng+rus')
                
                # Ищем числа в контексте (например, "14 октября" или просто "14")
                # Паттерны для поиска дня
                patterns = [
                    r'\b(\d{1,2})\s*(?:октября|october|окт)',  # "14 октября"
                    r'(?:октября|october|окт)\s*(\d{1,2})\b',  # "октября 14"
                    r'\b(\d{1,2})\b',  # Просто число
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1]
                        try:
                            num = int(match)
                            if 1 <= num <= 31:
                                day_candidates.append(num)
                                logger.debug(f"Method 2.{config}: found day {num}")
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Method 2 error with config {config}: {e}")
        
        # Метод 3: Использование image_to_data для точного поиска
        try:
            processed = _preprocess_for_ocr(top_third)
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, lang='eng')
            
            for i, text in enumerate(data['text']):
                if text and text.strip().isdigit():
                    num = int(text.strip())
                    if 1 <= num <= 31:
                        conf = int(data['conf'][i])
                        if conf > 30:  # Минимальная уверенность
                            day_candidates.append(num)
                            logger.debug(f"Method 3: found day {num} with confidence {conf}")
        except Exception as e:
            logger.debug(f"Method 3 error: {e}")
        
        # Выбираем наиболее часто встречающийся день
        if day_candidates:
            from collections import Counter
            counter = Counter(day_candidates)
            most_common = counter.most_common(1)[0][0]
            logger.info(f"Day candidates: {day_candidates}, selected: {most_common}")
            return most_common
        
        return None
    
    def process_image(self, image_input: Any) -> List[Dict]:
        """Обработка одного изображения."""
        img = _read_image(image_input)
        if img is None:
            logger.error("Cannot read image")
            return []
        
        logger.info(f"Processing image with shape: {img.shape}")
        
        # 1. Извлекаем день с улучшенным методом
        day = self._extract_day_advanced(img)
        
        if day is None:
            # Fallback: пробуем использовать текущий день
            current_day = datetime.now().day
            if 1 <= current_day <= 31:
                logger.warning(f"Day not detected, using current day: {current_day}")
                day = current_day
            else:
                logger.warning("Day not detected, using default: 14")
                day = 14  # Более разумный default для октября
        else:
            logger.info(f"Successfully detected day: {day}")
        
        # 2. Формируем дату
        try:
            slot_date = date(FIXED_YEAR, FIXED_MONTH, day)
            iso_date = slot_date.isoformat()
            logger.info(f"Using date: {iso_date}")
        except:
            logger.error(f"Invalid day: {day}")
            return []
        
        # 3. Ищем временные слоты
        all_ranges = []
        
        # Фокусируемся на нижней части изображения (где обычно слоты)
        h = img.shape[0]
        bottom_part = img[h//3:, :]
        
        # Метод 1: Простой OCR
        try:
            pil = Image.fromarray(cv2.cvtColor(bottom_part, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil, lang='eng+rus')
            ranges = _extract_time_ranges_robust(text)
            if ranges:
                all_ranges.extend(ranges)
                logger.info(f"Found {len(ranges)} time ranges with simple OCR")
        except Exception as e:
            logger.debug(f"Simple OCR failed: {e}")
        
        # Метод 2: С предобработкой
        if not all_ranges:
            try:
                processed = _preprocess_for_ocr(bottom_part)
                pil = Image.fromarray(processed)
                text = pytesseract.image_to_string(pil, lang='eng')
                ranges = _extract_time_ranges_robust(text)
                if ranges:
                    all_ranges.extend(ranges)
                    logger.info(f"Found {len(ranges)} time ranges with preprocessing")
            except Exception as e:
                logger.debug(f"Preprocessed OCR failed: {e}")
        
        # Формируем результат
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
                logger.info(f"Added slot: {iso_date} {start}-{end}")
        
        logger.info(f"Total found {len(slots)} unique slot(s)")
        return slots


# --------- Класс MemorySlotParser для совместимости ---------
class MemorySlotParser:
    """Парсер для работы со скриншотами из памяти."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.parser = NewFormatSlotParser(debug=debug)
        
        # Все атрибуты для совместимости
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
        """Обработка одного скриншота из памяти."""
        try:
            logger.info(f"Processing screenshot from memory, is_last={is_last}")
            
            slots = self.parser.process_image(image_bytes)
            
            self.accumulated_slots.extend(slots)
            self.screenshots_count += 1
            
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
        """Предобработка изображения."""
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
    
    def parse_time_in_text(self, text: str) -> Optional[Tuple[str, str]]:
        """Поиск времени в тексте."""
        ranges = _extract_time_ranges_robust(text)
        return ranges[0] if ranges else None
    
    def _extract_lines_from_data(self, data: Dict) -> List[Dict]:
        """Извлечение строк из OCR данных."""
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


class SlotParser:
    """Класс для обработки папки со скриншотами."""
    
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.parser = NewFormatSlotParser(debug=debug)
        self.cancelled_count = 0
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