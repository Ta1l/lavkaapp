# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Улучшенная версия с робастным определением дня из желтого квадрата.
"""

import os
import re
import logging
from datetime import date
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO
from collections import Counter

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

# --------- Константы и конфигурация ---------
FIXED_YEAR = 2025
FIXED_MONTH = 10

# Конфигурация поиска желтого квадрата
YELLOW_HSV_LOWER = np.array([8, 60, 80], dtype=np.uint8)
YELLOW_HSV_UPPER = np.array([50, 255, 255], dtype=np.uint8)
MIN_AREA_RATIO = 0.0005  # Минимальная площадь контура относительно изображения
MAX_AREA_RATIO = 0.1     # Максимальная площадь (чтобы не захватить весь экран)
ASPECT_RATIO_MIN = 0.5    # Минимальное соотношение сторон
ASPECT_RATIO_MAX = 2.5    # Максимальное соотношение сторон
FILL_RATIO_MIN = 0.4      # Минимальная заполненность контура

# Конфигурация OCR
OCR_CONFIDENCE_THRESHOLD = 50  # Минимальная уверенность OCR
OCR_SCALE_FACTOR = 3.5         # Масштабирование для OCR
PADDING_RATIO = 0.08           # Отступ вокруг желтого квадрата (% от размера)

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


def _crop_top(img: np.ndarray, crop_percent: float = 0.03) -> np.ndarray:
    """Обрезает верхнюю часть изображения (системную панель)."""
    h, w = img.shape[:2]
    crop_height = int(h * crop_percent)
    return img[crop_height:, :]


def _find_yellow_box_robust(img: np.ndarray, debug: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Робастный поиск желтого квадрата на всем изображении.
    Использует адаптивные параметры относительно размера изображения.
    """
    try:
        h, w = img.shape[:2]
        img_area = h * w
        
        # HSV маска для желтого/оранжевого
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
        
        # Дополнительная маска через Lab цветовое пространство
        # В Lab желтый имеет высокий b-канал
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Желтый в Lab: высокий b (>145) и средний/высокий L
        mask_lab = cv2.inRange(b_channel, 145, 255)
        
        # Объединяем маски
        mask = cv2.bitwise_or(mask_hsv, mask_lab)
        
        # Адаптивный размер ядра для морфологии
        kernel_size = max(3, int(round(min(h, w) * 0.008)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Морфологические операции для удаления шума и заполнения дыр
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        if debug:
            cv2.imwrite("debug_yellow_mask.png", mask)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No yellow contours found")
            return None
        
        # Фильтруем и оцениваем контуры
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Проверка размера относительно изображения
            area_ratio = area / img_area
            if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Проверка соотношения сторон
            aspect_ratio = w / float(h)
            if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                continue
            
            # Проверка заполненности
            rect_area = w * h
            fill_ratio = area / rect_area
            if fill_ratio < FILL_RATIO_MIN:
                continue
            
            # Добавляем кандидата с его площадью для сортировки
            candidates.append((x, y, w, h, area))
            
        if not candidates:
            logger.warning("No valid yellow boxes found after filtering")
            return None
        
        # Выбираем самый большой по площади контур
        candidates.sort(key=lambda c: c[4], reverse=True)
        best_box = candidates[0][:4]
        
        logger.info(f"Found yellow box at ({best_box[0]}, {best_box[1]}) size ({best_box[2]}x{best_box[3]})")
        
        if debug:
            debug_img = img.copy()
            x, y, w, h = best_box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite("debug_yellow_box_detected.png", debug_img)
        
        return best_box
        
    except Exception as e:
        logger.error(f"Error in _find_yellow_box_robust: {e}")
        return None


def _extract_day_from_box_advanced(img: np.ndarray, box: Tuple[int, int, int, int], debug: bool = False) -> Optional[int]:
    """
    Продвинутое извлечение дня из желтого квадрата.
    Использует image_to_data для выбора самого крупного числа.
    """
    try:
        x, y, w, h = box
        
        # Адаптивный отступ относительно размера бокса
        padding = max(4, int(round(PADDING_RATIO * max(w, h))))
        
        # Вырезаем ROI с отступом
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
        
        # Применяем CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Легкое размытие перед бинаризацией
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Масштабируем для лучшего OCR
        scale = OCR_SCALE_FACTOR
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        gray_scaled = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Список кандидатов (число, высота_bbox, площадь_bbox, уверенность)
        all_candidates = []
        
        # Пробуем разные варианты бинаризации и PSM режимы
        for invert in [True, False]:
            for psm in [6, 7, 8]:  # Разные режимы сегментации
                try:
                    # Бинаризация
                    if invert:
                        _, thresh = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    else:
                        _, thresh = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    if debug:
                        cv2.imwrite(f"debug_thresh_inv{invert}_psm{psm}.png", thresh)
                    
                    # OCR с image_to_data для получения bbox каждого элемента
                    config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789 -c tessedit_char_blacklist=/"
                    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Анализируем каждый распознанный элемент
                    for i in range(len(data["text"])):
                        text = str(data["text"][i]).strip()
                        conf = int(data["conf"][i]) if str(data["conf"][i]).lstrip('-').isdigit() else -1
                        
                        # Пропускаем элементы с низкой уверенностью или пустые
                        if not text or conf < OCR_CONFIDENCE_THRESHOLD:
                            continue
                        
                        # Проверяем, что это число от 1 до 31
                        if re.fullmatch(r"\d{1,2}", text):
                            val = int(text)
                            if 1 <= val <= 31:
                                # Получаем размеры bbox
                                bbox_width = int(data["width"][i])
                                bbox_height = int(data["height"][i])
                                bbox_area = bbox_width * bbox_height
                                
                                # Добавляем кандидата
                                all_candidates.append((val, bbox_height, bbox_area, conf, psm, invert))
                                
                                if debug:
                                    logger.debug(f"Candidate: day={val}, height={bbox_height}, area={bbox_area}, conf={conf}, psm={psm}, inv={invert}")
                
                except Exception as e:
                    logger.debug(f"OCR attempt failed (psm={psm}, invert={invert}): {e}")
                    continue
        
        if not all_candidates:
            logger.warning("No day candidates found with image_to_data")
            
            # Fallback: простой image_to_string
            for invert in [True, False]:
                try:
                    if invert:
                        _, thresh = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    else:
                        _, thresh = cv2.threshold(gray_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    config = "--psm 8 -c tessedit_char_whitelist=0123456789"
                    text = pytesseract.image_to_string(thresh, config=config)
                    
                    # Ищем числа, но берем только те, что в диапазоне 1-31
                    numbers = re.findall(r'\d+', text)
                    for num_str in numbers:
                        if len(num_str) <= 2:  # Игнорируем длинные числа
                            num = int(num_str)
                            if 1 <= num <= 31:
                                # Для fallback используем фиктивные размеры
                                all_candidates.append((num, 100, 10000, 75, 0, invert))
                                logger.info(f"Fallback found day: {num}")
                                
                except Exception as e:
                    logger.debug(f"Fallback failed (invert={invert}): {e}")
        
        if all_candidates:
            # Сортируем кандидатов:
            # 1. По высоте bbox (больше = лучше)
            # 2. По площади bbox (больше = лучше)  
            # 3. По уверенности (выше = лучше)
            all_candidates.sort(key=lambda c: (c[1], c[2], c[3]), reverse=True)
            
            best_candidate = all_candidates[0]
            day = best_candidate[0]
            
            logger.info(f"Selected day {day} from {len(all_candidates)} candidates")
            logger.debug(f"Best candidate details: day={day}, height={best_candidate[1]}, area={best_candidate[2]}, conf={best_candidate[3]}")
            
            # Дополнительная проверка: если есть несколько кандидатов с одинаковым числом,
            # это повышает уверенность
            day_counts = Counter([c[0] for c in all_candidates])
            if day_counts[day] > 1:
                logger.info(f"Day {day} appeared {day_counts[day]} times - high confidence")
            
            return day
        
        logger.warning("No valid day found in yellow box")
        return None
        
    except Exception as e:
        logger.error(f"Error in _extract_day_from_box_advanced: {e}")
        return None


def _find_time_slots(img: np.ndarray) -> List[Tuple[str, str]]:
    """
    Находит временные слоты на изображении.
    Фокусируется на нижней части где обычно расположены слоты.
    """
    try:
        # Берем нижние 3/4 изображения
        h = img.shape[0]
        bottom_part = img[h//4:, :]
        
        # Конвертируем в grayscale
        gray = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2GRAY)
        
        # Применяем OCR
        text = pytesseract.image_to_string(gray, lang='rus+eng')
        
        # Нормализуем текст
        text = text.replace('О', '0').replace('о', '0')  # Русские О на нули
        text = text.replace('З', '3').replace('з', '3')  # Русские З на 3
        text = text.replace('б', '6').replace('Б', '6')  # Русские Б на 6
        
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
        
        # Если не нашли слоты, пробуем с предобработкой
        if not slots:
            # CLAHE для улучшения контраста
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Бинаризация
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Повторяем OCR
            text = pytesseract.image_to_string(binary, lang='rus+eng')
            text = text.replace('О', '0').replace('о', '0')
            text = text.replace('З', '3').replace('з', '3')
            
            for pattern in TIME_RANGE_PATTERNS:
                matches = pattern.findall(text)
                for match in matches:
                    try:
                        h1, m1, h2, m2 = match
                        h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
                        
                        if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                            start_time = f"{h1:02d}:{m1:02d}"
                            end_time = f"{h2:02d}:{m2:02d}"
                            
                            if h2 > h1 or (h2 == h1 and m2 > m1):
                                slots.append((start_time, end_time))
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
        
        # 2. Обрезаем верх (системное время)
        img = _crop_top(img, crop_percent=0.03)
        
        # 3. Находим желтый квадрат (робастный поиск по всему изображению)
        yellow_box = _find_yellow_box_robust(img, debug=self.debug)
        
        if yellow_box is None:
            logger.error("Yellow box not found on image")
            return []
        
        # 4. Извлекаем день из желтого квадрата (продвинутый метод)
        day = _extract_day_from_box_advanced(img, yellow_box, debug=self.debug)
        
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
            logger.error(f"Error processing screenshot: {e}", exc_info=True)
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
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
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