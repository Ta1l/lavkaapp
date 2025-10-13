# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Версия с улучшенным извлечением дня из желтого квадрата.
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
    Улучшенное извлечение дня из желтого квадрата.
    Принимает: img (BGR), box (x, y, w, h).
    Возвращает: int (1..31) или None.
    Подход:
     - Crop + padding
     - LAB -> L канал, bilateral filter, CLAHE
     - adaptive threshold (несколько масштабов)
     - попытки OCR с разными PSM и whitelist
     - сегментация контуров: выбираем крупные символы (игнорируем мелкие после '/')
     - комбинируем кандидаты и выбираем наиболее частый/логичный
    """
    try:
        x, y, w, h = box

        # Padding, но не слишком большой
        padding = max(6, int(min(w, h) * 0.15))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            logger.warning("Empty ROI for yellow box")
            return None

        # Преобразование в LAB -> берем L (яркость)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Сглаживание, сохраняя края
        l = cv2.bilateralFilter(l, d=9, sigmaColor=75, sigmaSpace=75)

        # CLAHE (локальное усиление контраста)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Нормализуем яркость и увеличим изображение для OCR
        def resize_for_ocr(img_gray, scale):
            w0 = max(20, int(img_gray.shape[1] * scale))
            h0 = max(20, int(img_gray.shape[0] * scale))
            return cv2.resize(img_gray, (w0, h0), interpolation=cv2.INTER_CUBIC)

        candidates: List[int] = []

        # Функция: запустить pytesseract с несколькими конфигами на заданном изображении
        def ocr_trials(img_gray):
            texts = []
            # Простая бинаризация (адаптивная)
            try:
                th_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
            except Exception:
                _, th_adapt = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Инверсия если фон тёмный (чтобы получить тёмный текст на светлом фоне)
            mean_val = int(np.mean(img_gray))
            if mean_val < 120:
                th_used = cv2.bitwise_not(th_adapt)
            else:
                th_used = th_adapt

            # Попробуем несколько PSM и whitelist
            configs = [
                '--psm 8 -c tessedit_char_whitelist=0123456789/',
                '--psm 7 -c tessedit_char_whitelist=0123456789/',
                '--psm 10 -c tessedit_char_whitelist=0123456789'  # single char
            ]

            for cfg in configs:
                try:
                    text = pytesseract.image_to_string(th_used, config=cfg)
                    if text:
                        texts.append(text.strip())
                except Exception as e:
                    logger.debug(f"OCR attempt failed: {e}")
            return texts, th_used

        # Попробовать на нескольких масштабах (2x, 3x, 4x)
        scales = [2.0, 3.0, 4.0]
        ocr_texts = []
        last_thresh = None
        for s in scales:
            gray_resized = resize_for_ocr(l, s)
            texts, th_used = ocr_trials(gray_resized)
            ocr_texts.extend(texts)
            last_thresh = th_used

        # Также используем прямой OCR на исходном усиленном изображении (без масштабирования)
        try:
            fallback_text = pytesseract.image_to_string(l, config='--psm 6 -c tessedit_char_whitelist=0123456789/')
            if fallback_text:
                ocr_texts.append(fallback_text.strip())
        except Exception:
            pass

        logger.debug(f"OCR raw candidates: {ocr_texts}")

        # 1) Прямой парсинг из полученных строк
        for raw in ocr_texts:
            if not raw:
                continue
            # Нормализуем пробелы и заменяем похожие буквы
            r = raw.replace(' ', '').replace('O', '0').replace('o', '0').replace('l', '1')
            # Если есть '/', берем часть слева (обычно день)
            if '/' in r:
                left = r.split('/')[0]
                m = re.findall(r'\d{1,2}', left)
                for mg in m:
                    try:
                        val = int(mg)
                        if 1 <= val <= 31:
                            candidates.append(val)
                    except:
                        continue
            else:
                m = re.findall(r'\d{1,2}', r)
                for mg in m:
                    try:
                        val = int(mg)
                        if 1 <= val <= 31:
                            candidates.append(val)
                    except:
                        continue

        # 2) Контурный анализ — выделяем крупные символы, чтобы отделить основной (день) от мелких (например '12' справа)
        try:
            if last_thresh is None:
                gray0 = l.copy()
                _, last_thresh = cv2.threshold(gray0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Найдём контуры на последнем пороговом изображении
            contours, _ = cv2.findContours(last_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = []
            for cnt in contours:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                area = rw * rh
                # Фильтруем слишком маленькие объекты
                if area < 20 or rw < 6 or rh < 8:
                    continue
                rects.append((rx, ry, rw, rh, area))
            if rects:
                # Сортируем по x (лево->право)
                rects.sort(key=lambda r: r[0])
                # Рассчитаем медиану высот — это ориентир «крупного шрифта»
                heights = [r[3] for r in rects]
                med_h = int(np.median(heights))
                # Выбираем те rects, высота которых >= 0.7 * median (крупные цифры)
                big_rects = [r for r in rects if r[3] >= 0.7 * med_h]
                # Если большие прямоугольники есть, пытаемся OCR по ним по очереди (слева направо)
                if big_rects:
                    big_rects.sort(key=lambda r: r[0])
                    # объединяем bbox первых 1-2 больших символов (на случай двузначного дня)
                    # берем левую группу: все последовательные прямоугольники, пока расстояние между ними небольшое
                    groups = []
                    current_group = [big_rects[0]]
                    for r in big_rects[1:]:
                        prev = current_group[-1]
                        # если следующий рядом (по x) — считаем частью той же группы
                        if r[0] - (prev[0] + prev[2]) < max(10, int(prev[2] * 0.8)):
                            current_group.append(r)
                        else:
                            groups.append(current_group)
                            current_group = [r]
                    groups.append(current_group)
                    # Проанализируем первые две группы (левые)
                    take_groups = groups[:2]
                    for g in take_groups:
                        gx1 = min([r[0] for r in g])
                        gy1 = min([r[1] for r in g])
                        gx2 = max([r[0] + r[2] for r in g])
                        gy2 = max([r[1] + r[3] for r in g])
                        # небольшой отступ внутри ROI
                        pad = 2
                        gx1 = max(0, gx1 - pad); gy1 = max(0, gy1 - pad)
                        gx2 = min(last_thresh.shape[1], gx2 + pad); gy2 = min(last_thresh.shape[0], gy2 + pad)
                        crop = last_thresh[gy1:gy2, gx1:gx2]
                        # увеличим и OCR для этой части
                        crop_up = resize_for_ocr(crop, 3.0)
                        try:
                            txt = pytesseract.image_to_string(crop_up, config='--psm 7 -c tessedit_char_whitelist=0123456789')
                            txt = txt.strip()
                            if txt:
                                nums = re.findall(r'\d{1,2}', txt)
                                for ns in nums:
                                    v = int(ns)
                                    if 1 <= v <= 31:
                                        candidates.append(v)
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"Contour analysis failed: {e}")

        logger.debug(f"Day numeric candidates before selection: {candidates}")

        # Выбор финального кандидата: наиболее частый, с приоритетом двузначных (10-31)
        if candidates:
            cnt = Counter(candidates)
            # Попробуем сначала взять наиболее частый двузначный (если есть)
            for num, _ in cnt.most_common():
                if 10 <= num <= 31:
                    logger.info(f"Selected day (two-digit priority): {num}; candidates: {candidates}")
                    return num
            # Иначе — наиболее частый любой допустимый
            most_common = cnt.most_common(1)[0][0]
            logger.info(f"Selected day: {most_common}; candidates: {candidates}")
            return most_common

        # Fallback: поиск по всему тексту (ещё одна попытка)
        try:
            final_text = pytesseract.image_to_string(l, config='--psm 6')
            nums = re.findall(r'\d{1,2}', final_text)
            for ns in nums:
                v = int(ns)
                if 1 <= v <= 31:
                    logger.info(f"Fallback OCR found day: {v}")
                    return v
        except Exception:
            pass

        logger.warning("No valid day found in yellow box after all attempts")
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
        
        # 3. Извлекаем день (улучшенный метод)
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