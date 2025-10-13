# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Каждый скриншот = отдельный день с желтым выделением даты.
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
    # HH:MM - HH:MM
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    # HH.MM - HH.MM
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    # HHMM - HHMM
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


def _crop_top(img: np.ndarray, crop_percent: float = 0.05) -> np.ndarray:
    """Обрезает верхнюю часть изображения (системную панель)."""
    h, w = img.shape[:2]
    crop_height = int(h * crop_percent)
    return img[crop_height:, :]


def _find_yellow_box(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Находит желтый квадрат на изображении.
    Возвращает (x, y, width, height) или None.
    """
    try:
        # Конвертируем в HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Определяем диапазоны для желтого/оранжевого цвета
        # Желтый в HSV примерно от 20 до 35
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        
        # Дополнительный диапазон для оранжевого
        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([20, 255, 255])
        
        # Создаем маски
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Объединяем маски
        mask = cv2.bitwise_or(mask_yellow, mask_orange)
        
        # Ищем только в верхней половине изображения
        h = mask.shape[0]
        mask[h//2:, :] = 0
        
        # Морфологические операции для удаления шума
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No yellow contours found")
            return None
        
        # Фильтруем контуры по размеру и форме
        valid_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Слишком маленький
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Проверяем соотношение сторон (должен быть примерно квадрат)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.6 or aspect_ratio > 1.7:
                continue
            
            # Проверяем заполненность (контур должен быть достаточно заполнен)
            rect_area = w * h
            fill_ratio = area / rect_area
            if fill_ratio < 0.5:
                continue
            
            valid_boxes.append((x, y, w, h, area))
        
        if not valid_boxes:
            logger.warning("No valid yellow boxes found")
            return None
        
        # Выбираем самый большой по площади
        valid_boxes.sort(key=lambda b: b[4], reverse=True)
        best_box = valid_boxes[0][:4]
        
        logger.info(f"Found yellow box at ({best_box[0]}, {best_box[1]}) size ({best_box[2]}x{best_box[3]})")
        return best_box
        
    except Exception as e:
        logger.error(f"Error finding yellow box: {e}")
        return None


def _extract_day_from_box(img: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[int]:
    """
    Извлекает день из желтого квадрата.
    """
    try:
        x, y, w, h = box
        
        # Вырезаем область с небольшим отступом
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        roi = img[y1:y2, x1:x2]
        
        if roi.size == 0:
            logger.error("Empty ROI")
            return None
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Увеличиваем для лучшего OCR
        scale = 3.0
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Применяем пороговую обработку
        # Пробуем инвертировать, так как цифра может быть темной на желтом фоне
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # OCR - ищем только цифры
        custom_config = r'--psm 8 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Извлекаем числа
        numbers = re.findall(r'\d+', text)
        
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 31:
                logger.info(f"Extracted day from yellow box: {num}")
                return num
        
        # Если не нашли с инверсией, пробуем без инверсии
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config=custom_config)
        numbers = re.findall(r'\d+', text)
        
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 31:
                logger.info(f"Extracted day from yellow box (non-inverted): {num}")
                return num
        
        logger.warning(f"No valid day found in yellow box, OCR result: {text}")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting day from box: {e}")
        return None


def _find_time_slots(img: np.ndarray) -> List[Tuple[str, str]]:
    """
    Находит временные слоты на изображении.
    """
    try:
        # Фокусируемся на нижней части изображения (где обычно слоты)
        h = img.shape[0]
        bottom_part = img[h//4:, :]
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2GRAY)
        
        # Применяем OCR
        text = pytesseract.image_to_string(gray, lang='rus+eng')
        
        # Нормализуем текст
        text = text.replace('О', '0').replace('о', '0')  # Русские О на нули
        text = text.replace('З', '3').replace('з', '3')  # Русские З на 3
        
        slots = []
        
        # Ищем временные диапазоны
        for pattern in TIME_RANGE_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    h1, m1, h2, m2 = match
                    h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
                    
                    # Валидация времени
                    if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                        start_time = f"{h1:02d}:{m1:02d}"
                        end_time = f"{h2:02d}:{m2:02d}"
                        
                        # Проверяем что время окончания больше времени начала
                        if h2 > h1 or (h2 == h1 and m2 > m1):
                            slots.append((start_time, end_time))
                            logger.debug(f"Found time slot: {start_time} - {end_time}")
                except:
                    continue
        
        # Убираем дубликаты
        unique_slots = []
        seen = set()
        for slot in slots:
            if slot not in seen:
                seen.add(slot)
                unique_slots.append(slot)
        
        return unique_slots
        
    except Exception as e:
        logger.error(f"Error finding time slots: {e}")
        return []


# --------- Основной класс парсера ---------
class NewFormatSlotParser:
    """Парсер для нового формата скриншотов с желтым выделением дня."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
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
        
        # 2. Обрезаем верх (системное время)
        img = _crop_top(img, crop_percent=0.05)
        
        # 3. Находим желтый квадрат
        yellow_box = _find_yellow_box(img)
        
        if yellow_box is None:
            logger.error("Yellow box not found on image")
            return []
        
        # 4. Извлекаем день из желтого квадрата
        day = _extract_day_from_box(img, yellow_box)
        
        if day is None:
            logger.error("Could not extract day from yellow box")
            return []
        
        # 5. Формируем дату
        try:
            slot_date = date(FIXED_YEAR, FIXED_MONTH, day)
            iso_date = slot_date.isoformat()
            logger.info(f"Date for slots: {iso_date}")
        except ValueError as e:
            logger.error(f"Invalid date: year={FIXED_YEAR}, month={FIXED_MONTH}, day={day}: {e}")
            return []
        
        # 6. Находим временные слоты
        time_slots = _find_time_slots(img)
        
        if not time_slots:
            logger.warning(f"No time slots found for {iso_date}")
            return []
        
        # 7. Формируем результат
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


# --------- Класс MemorySlotParser для обработки из памяти ---------
class MemorySlotParser:
    """Парсер для обработки скриншотов из памяти (BytesIO)."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.parser = NewFormatSlotParser(debug=debug)
        self.accumulated_slots = []
        self.screenshots_count = 0
        
        # Для совместимости со старым кодом
        self.base_path = ""
        self.cancelled_count = 0
        self.last_error = None
        
    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """
        Обрабатывает скриншот из памяти.
        Если is_last=True, возвращает все накопленные слоты.
        """
        try:
            logger.info(f"Processing screenshot #{self.screenshots_count + 1}, is_last={is_last}")
            
            # Обрабатываем изображение
            slots = self.parser.process_image(image_bytes)
            
            # Добавляем к накопленным
            if slots:
                self.accumulated_slots.extend(slots)
                logger.info(f"Added {len(slots)} slots from screenshot #{self.screenshots_count + 1}")
            else:
                logger.warning(f"No slots found in screenshot #{self.screenshots_count + 1}")
            
            self.screenshots_count += 1
            
            # Если последний скриншот, возвращаем все уникальные слоты
            if is_last:
                return self.get_all_slots()
            
            return slots
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error processing screenshot: {e}")
            return []
    
    def process_screenshot(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """Алиас для process_screenshot_from_memory."""
        return self.process_screenshot_from_memory(image_bytes, is_last)
    
    def process_image(self, image_input: Any) -> List[Dict]:
        """Обрабатывает одно изображение."""
        try:
            return self.parser.process_image(image_input)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error processing image: {e}")
            return []
    
    def process_image_bytes(self, image_bytes: bytes) -> List[Dict]:
        """Обрабатывает изображение из байтов."""
        try:
            return self.parser.process_image(image_bytes)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error processing image bytes: {e}")
            return []
    
    def get_all_slots(self) -> List[Dict]:
        """Возвращает все уникальные накопленные слоты."""
        if not self.accumulated_slots:
            logger.warning("No accumulated slots to return")
            return []
        
        # Убираем дубликаты
        seen = set()
        unique = []
        for slot in self.accumulated_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key)
                unique.append(slot)
        
        # Сортируем по дате и времени
        unique.sort(key=lambda s: (s["date"], s["startTime"]))
        
        logger.info(f"Returning {len(unique)} unique slots from {self.screenshots_count} screenshots")
        return unique
    
    def clear(self):
        """Очищает накопленные данные."""
        self.accumulated_slots = []
        self.screenshots_count = 0
        self.last_error = None
        self.cancelled_count = 0
        logger.info("Cleared accumulated data")
    
    def reset(self):
        """Алиас для clear."""
        self.clear()
    
    # Методы для совместимости
    def preprocess_image_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Для совместимости со старым кодом."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        return gray
    
    def _extract_lines_from_data(self, data: Dict) -> List[Dict]:
        """Для совместимости со старым кодом."""
        return []
    
    def process_all_screenshots(self) -> List[Dict]:
        """Для совместимости."""
        return self.get_all_slots()


# --------- Класс SlotParser для обработки папки ---------
class SlotParser:
    """Парсер для обработки папки со скриншотами."""
    
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.parser = NewFormatSlotParser(debug=debug)
        self.cancelled_count = 0
        
    def process_all_screenshots(self) -> List[Dict]:
        """Обрабатывает все скриншоты в папке."""
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}")
            return []
        
        # Находим все изображения
        files = []
        for f in os.listdir(self.base_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(self.base_path, f))
        
        if not files:
            logger.warning(f"No images found in: {self.base_path}")
            return []
        
        # Сортируем по времени создания
        files.sort(key=lambda x: os.path.getctime(x))
        
        logger.info(f"Found {len(files)} images to process")
        
        all_slots = []
        
        # Обрабатываем каждый файл
        for i, filepath in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {os.path.basename(filepath)}")
            try:
                slots = self.parser.process_image(filepath)
                if slots:
                    all_slots.extend(slots)
                    logger.info(f"Found {len(slots)} slots in {os.path.basename(filepath)}")
                else:
                    logger.warning(f"No slots found in {os.path.basename(filepath)}")
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        # Убираем дубликаты
        seen = set()
        unique = []
        for slot in all_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key)
                unique.append(slot)
        
        # Сортируем
        unique.sort(key=lambda s: (s["date"], s["startTime"]))
        
        logger.info(f"Total: {len(unique)} unique slots from {len(files)} files")
        return unique


# Экспорт классов
__all__ = ["SlotParser", "MemorySlotParser", "NewFormatSlotParser"]