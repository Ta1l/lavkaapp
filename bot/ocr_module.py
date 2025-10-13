# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Упрощенная и надежная версия.
"""

import os
import re
import logging
from datetime import date
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

# Паттерны для поиска времени
TIME_RANGE_PATTERNS = [
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    re.compile(r"(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})"),
]


# --------- Вспомогательные функции ---------
def _read_image(image_input: Any) -> Optional[np.ndarray]:
    """Читает изображение из разных источников."""
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.error(f"File not found: {image_input}")
                return None
            img = cv2.imread(image_input)
            return img
        
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        
        if isinstance(image_input, BytesIO):
            image_input.seek(0)
            data = image_input.read()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        
        if isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        logger.error(f"Unknown image type: {type(image_input)}")
        return None
        
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return None


def _find_yellow_box(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Находит желтый квадрат на изображении.
    Возвращает (x, y, width, height) или None.
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Расширенный диапазон для желтого/оранжевого
        lower = np.array([10, 50, 50])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Ищем в верхней половине изображения
        h = mask.shape[0]
        mask[h//2:, :] = 0
        
        # Морфология для очистки
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Фильтруем по размеру и форме
        valid_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # Квадратная форма
            if 0.7 <= aspect_ratio <= 1.5:
                valid_boxes.append((x, y, w, h, area))
        
        if not valid_boxes:
            return None
        
        # Берем самый большой
        valid_boxes.sort(key=lambda b: b[4], reverse=True)
        best = valid_boxes[0][:4]
        
        logger.info(f"Found yellow box at ({best[0]}, {best[1]}) size ({best[2]}x{best[3]})")
        return best
        
    except Exception as e:
        logger.error(f"Error finding yellow box: {e}")
        return None


def _extract_day_simple(img: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[int]:
    """
    Упрощенное извлечение дня из желтого квадрата.
    Использует несколько методов OCR и выбирает наиболее вероятный результат.
    """
    try:
        x, y, w, h = box
        
        # Вырезаем с небольшим отступом
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Увеличиваем изображение
        scale = 3.0
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        candidates = []
        
        # Метод 1: Простой OCR с инверсией (темный текст на светлом фоне)
        _, thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh_inv, config='--psm 8 -c tessedit_char_whitelist=0123456789')
        logger.debug(f"OCR (inverted): '{text.strip()}'")
        
        # Извлекаем все числа
        numbers = re.findall(r'\d+', text)
        for num_str in numbers:
            # Если число длинное (например, "412" из "4/12"), берем первые 1-2 цифры
            if len(num_str) > 2:
                # Пробуем первые две цифры
                if len(num_str) >= 2:
                    test_num = int(num_str[:2])
                    if 1 <= test_num <= 31:
                        candidates.append(test_num)
                # И первую цифру
                test_num = int(num_str[0])
                if 1 <= test_num <= 31:
                    candidates.append(test_num)
            else:
                num = int(num_str)
                if 1 <= num <= 31:
                    candidates.append(num)
        
        # Метод 2: OCR без инверсии
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config='--psm 8 -c tessedit_char_whitelist=0123456789')
        logger.debug(f"OCR (normal): '{text.strip()}'")
        
        numbers = re.findall(r'\d+', text)
        for num_str in numbers:
            if len(num_str) > 2:
                if len(num_str) >= 2:
                    test_num = int(num_str[:2])
                    if 1 <= test_num <= 31:
                        candidates.append(test_num)
                test_num = int(num_str[0])
                if 1 <= test_num <= 31:
                    candidates.append(test_num)
            else:
                num = int(num_str)
                if 1 <= num <= 31:
                    candidates.append(num)
        
        # Метод 3: PSM 7 (одна строка текста)
        text = pytesseract.image_to_string(thresh_inv, config='--psm 7 -c tessedit_char_whitelist=0123456789/')
        logger.debug(f"OCR (psm 7): '{text.strip()}'")
        
        # Если есть слэш, берем число до него
        if '/' in text:
            parts = text.split('/')
            if parts[0]:
                numbers = re.findall(r'\d+', parts[0])
                for num_str in numbers:
                    if len(num_str) <= 2:
                        num = int(num_str)
                        if 1 <= num <= 31:
                            candidates.append(num)
        else:
            numbers = re.findall(r'\d+', text)
            for num_str in numbers:
                if len(num_str) <= 2:
                    num = int(num_str)
                    if 1 <= num <= 31:
                        candidates.append(num)
        
        # Метод 4: CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh_enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        text = pytesseract.image_to_string(thresh_enhanced, config='--psm 8')
        logger.debug(f"OCR (CLAHE): '{text.strip()}'")
        
        # Более гибкий поиск чисел
        # Ищем паттерны вида "25", "25/", "25 ", но не "4/12"
        patterns = [
            r'^(\d{1,2})\s*[/\s]',  # Число в начале с разделителем
            r'^(\d{1,2})$',         # Просто число
            r'(\d{1,2})\s+\d',      # Число с пробелом перед другим числом
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.strip())
            if match:
                num = int(match.group(1))
                if 1 <= num <= 31:
                    candidates.append(num)
        
        if candidates:
            # Выбираем наиболее часто встречающееся число
            from collections import Counter
            counter = Counter(candidates)
            most_common = counter.most_common(1)[0][0]
            logger.info(f"Day candidates: {candidates}, selected: {most_common}")
            return most_common
        
        # Последняя попытка: весь текст без ограничений
        text = pytesseract.image_to_string(gray)
        logger.debug(f"OCR (fallback): '{text.strip()}'")
        
        # Ищем двузначные числа от 10 до 31 (они точно не могут быть частью "4/12")
        numbers = re.findall(r'\b([12][0-9]|3[01])\b', text)
        if numbers:
            num = int(numbers[0])
            logger.info(f"Fallback found day: {num}")
            return num
        
        # Ищем однозначные
        numbers = re.findall(r'\b([1-9])\b', text)
        if numbers:
            num = int(numbers[0])
            logger.info(f"Fallback found day: {num}")
            return num
        
        logger.warning("No valid day found in yellow box")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting day: {e}")
        return None


def _find_time_slots(img: np.ndarray) -> List[Tuple[str, str]]:
    """
    Находит временные слоты на изображении.
    """
    try:
        # Берем нижние 3/4 изображения
        h = img.shape[0]
        bottom_part = img[h//4:, :]
        
        gray = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2GRAY)
        
        # OCR с русским и английским
        text = pytesseract.image_to_string(gray, lang='rus+eng')
        
        # Нормализация
        text = text.replace('О', '0').replace('о', '0')
        text = text.replace('З', '3').replace('з', '3')
        
        slots = []
        
        # Поиск временных диапазонов
        for pattern in TIME_RANGE_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    h1, m1, h2, m2 = match
                    h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
                    
                    if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                        if h2 > h1 or (h2 == h1 and m2 > m1):
                            start_time = f"{h1:02d}:{m1:02d}"
                            end_time = f"{h2:02d}:{m2:02d}"
                            slots.append((start_time, end_time))
                            logger.debug(f"Found slot: {start_time} - {end_time}")
                except:
                    continue
        
        # Убираем дубликаты
        unique = []
        seen = set()
        for slot in slots:
            if slot not in seen:
                seen.add(slot)
                unique.append(slot)
        
        return unique
        
    except Exception as e:
        logger.error(f"Error finding time slots: {e}")
        return []


# --------- Основной класс парсера ---------
class NewFormatSlotParser:
    """Парсер для скриншотов с желтым выделением дня."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
        
    def process_image(self, image_input: Any) -> List[Dict]:
        """
        Обрабатывает одно изображение и возвращает найденные слоты.
        """
        # 1. Читаем изображение
        img = _read_image(image_input)
        if img is None:
            logger.error("Failed to read image")
            return []
        
        logger.info(f"Processing image, shape: {img.shape}")
        
        # 2. Находим желтый квадрат
        yellow_box = _find_yellow_box(img)
        
        if yellow_box is None:
            logger.error("Yellow box not found")
            return []
        
        # 3. Извлекаем день (упрощенный метод)
        day = _extract_day_simple(img, yellow_box)
        
        if day is None:
            logger.error("Could not extract day from yellow box")
            # Пробуем fallback - текущий день
            from datetime import datetime
            day = datetime.now().day
            if day > 31 or day < 1:
                day = 1
            logger.warning(f"Using fallback day: {day}")
        
        # 4. Формируем дату
        try:
            slot_date = date(FIXED_YEAR, FIXED_MONTH, day)
            iso_date = slot_date.isoformat()
            logger.info(f"Date for slots: {iso_date}")
        except ValueError as e:
            logger.error(f"Invalid date: {e}")
            return []
        
        # 5. Находим временные слоты
        time_slots = _find_time_slots(img)
        
        if not time_slots:
            logger.warning(f"No time slots found for {iso_date}")
            return []
        
        # 6. Формируем результат
        result = []
        for start_time, end_time in time_slots:
            slot = {
                "date": iso_date,
                "startTime": start_time,
                "endTime": end_time,
                "assignToSelf": True
            }
            result.append(slot)
            logger.info(f"Found slot: {iso_date} {start_time}-{end_time}")
        
        return result


# --------- Класс MemorySlotParser ---------
class MemorySlotParser:
    """Парсер для обработки скриншотов из памяти."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.parser = NewFormatSlotParser(debug=debug)
        self.accumulated_slots = []
        self.screenshots_count = 0
        
        # Для совместимости
        self.base_path = ""
        self.cancelled_count = 0
        self.last_error = None
        
    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """
        Обрабатывает скриншот из памяти.
        """
        try:
            logger.info(f"Processing screenshot #{self.screenshots_count + 1}, is_last={is_last}")
            
            slots = self.parser.process_image(image_bytes)
            
            if slots:
                self.accumulated_slots.extend(slots)
                logger.info(f"Added {len(slots)} slots from screenshot #{self.screenshots_count + 1}")
            else:
                logger.warning(f"No slots found in screenshot #{self.screenshots_count + 1}")
            
            self.screenshots_count += 1
            
            if is_last:
                return self.get_all_slots()
            
            return slots
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error: {e}")
            return []
    
    def process_screenshot(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """Алиас."""
        return self.process_screenshot_from_memory(image_bytes, is_last)
    
    def process_image(self, image_input: Any) -> List[Dict]:
        """Обработка одного изображения."""
        try:
            return self.parser.process_image(image_input)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error: {e}")
            return []
    
    def process_image_bytes(self, image_bytes: bytes) -> List[Dict]:
        """Обработка из байтов."""
        try:
            return self.parser.process_image(image_bytes)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error: {e}")
            return []
    
    def get_all_slots(self) -> List[Dict]:
        """Возвращает все уникальные слоты."""
        if not self.accumulated_slots:
            logger.warning("No accumulated slots to return")
            return []
        
        seen = set()
        unique = []
        for slot in self.accumulated_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key)
                unique.append(slot)
        
        unique.sort(key=lambda s: (s["date"], s["startTime"]))
        
        logger.info(f"Returning {len(unique)} unique slots from {self.screenshots_count} screenshots")
        return unique
    
    def clear(self):
        """Очистка."""
        self.accumulated_slots = []
        self.screenshots_count = 0
        self.last_error = None
        self.cancelled_count = 0
    
    def reset(self):
        """Алиас для clear."""
        self.clear()
    
    # Методы для совместимости
    def preprocess_image_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        return gray
    
    def _extract_lines_from_data(self, data: Dict) -> List[Dict]:
        return []
    
    def process_all_screenshots(self) -> List[Dict]:
        return self.get_all_slots()


# --------- Класс SlotParser ---------
class SlotParser:
    """Парсер для папки со скриншотами."""
    
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.parser = NewFormatSlotParser(debug=debug)
        self.cancelled_count = 0
        
    def process_all_screenshots(self) -> List[Dict]:
        """Обрабатывает все скриншоты в папке."""
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}")
            return []
        
        files = []
        for f in os.listdir(self.base_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(self.base_path, f))
        
        if not files:
            logger.warning(f"No images found in: {self.base_path}")
            return []
        
        files.sort(key=lambda x: os.path.getctime(x))
        
        logger.info(f"Found {len(files)} images")
        
        all_slots = []
        
        for i, filepath in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {os.path.basename(filepath)}")
            try:
                slots = self.parser.process_image(filepath)
                if slots:
                    all_slots.extend(slots)
            except Exception as e:
                logger.error(f"Error: {e}")
        
        seen = set()
        unique = []
        for slot in all_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key)
                unique.append(slot)
        
        unique.sort(key=lambda s: (s["date"], s["startTime"]))
        
        logger.info(f"Total: {len(unique)} unique slots")
        return unique


__all__ = ["SlotParser", "MemorySlotParser", "NewFormatSlotParser"]