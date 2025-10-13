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
    Надёжное извлечение дня из жёлтого квадрата (оптимизировано для чёрного шрифта на жёлтом фоне).
    Возвращает int 1..31 или None.
    Подход:
      - crop с padding
      - несколько масок (HSV-жёлтый/не-жёлтый, V-канал инвертированный)
      - морфология + дилатация для усиления тонких штрихов
      - OCR через image_to_data на каждом варианте
      - агрегация кандидатов по суммарной confidence
    """
    try:
        x, y, w, h = box
        # padding ~15% но не слишком большой
        padding = max(6, int(min(w, h) * 0.15))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        roi_bgr = img[y1:y2, x1:x2].copy()
        if roi_bgr.size == 0:
            logger.warning("Empty ROI for yellow box")
            return None

        # --- Подготовка нескольких предобработок ---
        preprocessed = {}

        # 1) L-CLAHE (LAB -> L)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.bilateralFilter(l, d=7, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        preprocessed["l_clahe"] = l_clahe

        # 2) V-channel inverted (чёрный текст -> белый на тёмном фоне)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        v_blur = cv2.GaussianBlur(v, (3, 3), 0)
        _, v_inv = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # усиление тонких штрихов
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        v_inv = cv2.morphologyEx(v_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
        v_inv = cv2.dilate(v_inv, kernel, iterations=1)
        preprocessed["v_inv"] = v_inv

        # 3) mask = dark pixels inside yellow region (цветовая сегментация)
        lower_y = np.array([10, 80, 80])
        upper_y = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_y, upper_y)
        non_yellow = cv2.bitwise_not(yellow_mask)
        # соединяем с v_inv чтобы убрать артефакты вокруг
        combined = cv2.bitwise_and(v_inv, non_yellow)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        preprocessed["combined"] = combined

        # 4) адаптивный порог от L-CLAHE
        try:
            th_adapt = cv2.adaptiveThreshold(l_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
            preprocessed["th_adapt"] = th_adapt
        except Exception:
            _, th_ot = cv2.threshold(l_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            preprocessed["th_adapt"] = th_ot

        # расширяем набор входов: разные масштабы
        scales = [1.5, 2.5, 3.5]

        # --- OCR и сбор кандидатов ---
        candidates = []  # list of (num:int, score:float, source:str)

        def collect_from_img(img_gray, source_name):
            # масштабируем для OCR (чем крупнее — тем лучше для мелкого UI)
            h0, w0 = img_gray.shape
            scale = max(1.0, min(4.0, 300.0 / max(h0, w0)))
            img_r = cv2.resize(img_gray, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_CUBIC)
            # пробуем несколько psm и whitelist
            cfgs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789/',
                '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            ]
            for cfg in cfgs:
                try:
                    # используем image_to_data — получаем confidence для каждого токена
                    data = pytesseract.image_to_data(img_r, config=cfg, output_type=pytesseract.Output.DICT)
                    n = len(data.get("text", []))
                    for i in range(n):
                        txt = (data["text"][i] or "").strip()
                        conf = int(data["conf"][i]) if str(data["conf"][i]).isdigit() else -1
                        if not txt:
                            continue
                        # нормализуем и ищем числа (особенно учитываем '/')
                        txt_norm = txt.replace(' ', '').replace('O', '0').replace('o','0').replace('l','1').replace('I','1')
                        if '/' in txt_norm:
                            left = txt_norm.split('/')[0]
                            m = re.findall(r'\d{1,2}', left)
                            for mg in m:
                                val = int(mg)
                                if 1 <= val <= 31:
                                    candidates.append((val, max(conf, 0), f"{source_name}[{cfg}]"))
                        else:
                            m = re.findall(r'\d{1,2}', txt_norm)
                            for mg in m:
                                val = int(mg)
                                if 1 <= val <= 31:
                                    candidates.append((val, max(conf, 0), f"{source_name}[{cfg}]"))
                except Exception as e:
                    logger.debug(f"OCR fail {source_name} cfg={cfg}: {e}")

        # direct runs on preprocessed images
        for name, im in preprocessed.items():
            # run on base and on scaled variants
            collect_from_img(im, name)
            for s in scales:
                h0, w0 = im.shape[:2]
                ir = cv2.resize(im, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_CUBIC)
                collect_from_img(ir, f"{name}_x{s}")

        # еще одна стратегия: connected components на combined, попытки OCR на каждой компоненте
        try:
            blobs_img = preprocessed.get("combined", preprocessed.get("v_inv"))
            if blobs_img is not None:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blobs_img, connectivity=8)
                for i in range(1, num_labels):
                    x_b, y_b, w_b, h_b, area_b = stats[i]
                    if area_b < 10 or w_b < 4 or h_b < 6:
                        continue
                    crop = blobs_img[y_b:y_b+h_b, x_b:x_b+w_b]
                    # pad slightly
                    pad = 2
                    crop_p = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
                    # invert to white background for some OCR configs
                    crop_inv = cv2.bitwise_not(crop_p)
                    # try OCR on crop_inv
                    collect_from_img(crop_inv, f"cc_{i}")
        except Exception:
            logger.debug("Connected components pass failed")

        # --- Выбор финального кандидата ---
        if candidates:
            # агрегируем по числу: суммируем confidence
            score_map = {}
            for num, conf, src in candidates:
                score_map.setdefault(num, 0)
                score_map[num] += max(0, conf)
            # отдаём приоритет двузначным (10-31), затем по суммарной confidence
            sorted_nums = sorted(score_map.items(), key=lambda kv: (- (10 <= kv[0] <= 31), -kv[1], -kv[0]))
            chosen_num = sorted_nums[0][0]
            logger.info(f"Day candidates (agg): {score_map}, selected: {chosen_num}")
            return chosen_num

        # Fallbacks: более агрессивный поиск в тексте без whitelist
        try:
            # попытка без ограничений на PSM
            for name, im in preprocessed.items():
                txt = pytesseract.image_to_string(im, config='--oem 3 --psm 6')
                if txt:
                    m = re.findall(r'\b([12]?\d|3[01])\b', txt)
                    if m:
                        v = int(m[0])
                        if 1 <= v <= 31:
                            logger.info(f"Fallback text found day: {v} from {name}")
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