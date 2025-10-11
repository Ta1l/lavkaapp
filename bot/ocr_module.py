# bot/ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для распознавания новых скриншотов смен.
Совместим со старой логикой (SlotParser / MemorySlotParser).
"""

import os
import re
import cv2
import pytesseract
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, date
from typing import List, Tuple, Optional
import logging

# Логгер
logger = logging.getLogger("lavka.ocr_module")
logger.setLevel(logging.INFO)

# Путь к tesseract (скорректируй при необходимости)
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

# Регулярка для поиска временных диапазонов
TIME_RANGE_RE = re.compile(r"(\d{1,2}[:.]\d{2})\s*[-–—]\s*(\d{1,2}[:.]\d{2})")


# -------------------------------------------------------------
# Основной класс для распознавания одного скриншота
# -------------------------------------------------------------
class NewFormatSlotParser:
    def __init__(self, now: Optional[datetime] = None):
        self.now = now or datetime.now()

    @staticmethod
    def _read_image(image_input) -> Optional[np.ndarray]:
        """Считывает изображение из пути, BytesIO или PIL.Image"""
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                return img
            if isinstance(image_input, BytesIO):
                image_input.seek(0)
                pil = Image.open(image_input).convert("RGB")
                arr = np.array(pil)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if isinstance(image_input, Image.Image):
                arr = np.array(image_input.convert("RGB"))
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return None
        except Exception as e:
            logger.error(f"_read_image error: {e}")
            return None

    @staticmethod
    def _preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
        """Подготовка изображения к OCR"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        scale = 1.4
        h, w = gray.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bw) < 127:
            bw = cv2.bitwise_not(bw)
        return bw

    @staticmethod
    def _find_yellow_region(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Ищет жёлтую область с датой"""
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            lower = np.array([12, 100, 100])
            upper = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:5]:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                return (x, y, w, h)
            return None
        except Exception as e:
            logger.error(f"_find_yellow_region error: {e}")
            return None

    @staticmethod
    def _ocr_day_number_from_roi(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """OCR-дата из жёлтого блока"""
        try:
            x, y, w, h = bbox
            pad_x = max(4, int(w * 0.12))
            pad_y = max(2, int(h * 0.12))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(bgr.shape[1], x + w + pad_x)
            y1 = min(bgr.shape[0], y + h + pad_y)
            roi = bgr[y0:y1, x0:x1].copy()
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
            config = "--psm 6 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thr, config=config, lang='eng+rus')
            digits = re.findall(r"\d{1,2}", text)
            if not digits:
                return None
            day = int(digits[0])
            if 1 <= day <= 31:
                return day
            return None
        except Exception as e:
            logger.error(f"_ocr_day_number_from_roi error: {e}")
            return None

    def _resolve_date_from_day(self, day: int) -> Optional[date]:
        """Определяет дату по номеру дня (учитывает переход месяца)"""
        today = self.now.date()
        candidates = []
        year = today.year
        month = today.month

        def try_make(y, m, d):
            try:
                return date(y, m, d)
            except Exception:
                return None

        candidates.append(try_make(year, month, day))
        for delta in (-1, 1, 2):
            m = month + delta
            y = year
            if m < 1:
                m += 12
                y -= 1
            elif m > 12:
                m -= 12
                y += 1
            candidates.append(try_make(y, m, day))

        candidates = [c for c in candidates if c]
        candidates = sorted(set(candidates))
        if not candidates:
            return None

        lookahead_days = 13
        best = None
        best_delta = 10**9

        for c in candidates:
            delta = (c - today).days
            if 0 <= delta <= lookahead_days:
                if delta < best_delta:
                    best = c
                    best_delta = delta

        if best is None:
            future_candidates = [(c, (c - today).days) for c in candidates if (c - today).days > 0]
            if future_candidates:
                return min(future_candidates, key=lambda x: x[1])[0]
            return min(candidates, key=lambda c: abs((c - today).days))

        return best

    def _extract_time_ranges(self, bgr: np.ndarray) -> List[Tuple[str, str]]:
        """Извлекает временные диапазоны"""
        try:
            proc = self._preprocess_for_ocr(bgr)
            pil = Image.fromarray(proc)
            config = "--psm 6 -c tessedit_char_whitelist=0123456789:.-–—"
            raw = pytesseract.image_to_string(pil, config=config, lang='eng+rus')
            if not raw:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pil2 = Image.fromarray(gray)
                raw = pytesseract.image_to_string(pil2, config=config, lang='eng+rus')

            matches = TIME_RANGE_RE.findall(raw)
            results = []
            for m in matches:
                t0 = m[0].replace(".", ":")
                t1 = m[1].replace(".", ":")
                def norm(t):
                    parts = t.split(":")
                    if len(parts) != 2:
                        return None
                    hh = int(parts[0])
                    mm = int(parts[1])
                    if 0 <= hh < 24 and 0 <= mm < 60:
                        return f"{hh:02d}:{mm:02d}"
                    return None
                n0 = norm(t0)
                n1 = norm(t1)
                if n0 and n1:
                    results.append((n0, n1))
            uniq = []
            seen = set()
            for s, e in results:
                key = (s, e)
                if key not in seen:
                    seen.add(key)
                    uniq.append((s, e))
            uniq.sort(key=lambda x: x[0])
            return uniq
        except Exception as e:
            logger.error(f"_extract_time_ranges error: {e}")
            return []

    def process_image(self, image_input) -> List[dict]:
        """Распознаёт один скриншот"""
        img = self._read_image(image_input)
        if img is None:
            logger.error("Cannot read image")
            return []

        bbox = self._find_yellow_region(img)
        day = None
        if bbox:
            day = self._ocr_day_number_from_roi(img, bbox)
        else:
            logger.info("Yellow region not found — fallback OCR")
            h = img.shape[0]
            top_strip = img[:int(h * 0.22), :]
            pil = Image.fromarray(self._preprocess_for_ocr(top_strip))
            config = "--psm 6 -c tessedit_char_whitelist=0123456789"
            txt = pytesseract.image_to_string(pil, config=config, lang='eng+rus')
            digits = re.findall(r"\d{1,2}", txt)
            if digits:
                day = int(digits[0])

        if not day:
            logger.error("Day number not detected")
            return []

        resolved = self._resolve_date_from_day(day)
        if not resolved:
            logger.error("Could not resolve date from day number")
            return []

        iso_date = resolved.isoformat()
        time_ranges = self._extract_time_ranges(img)
        if not time_ranges:
            logger.info("No time ranges found")
            return []

        slots = []
        seen = set()
        for s, e in time_ranges:
            key = (iso_date, s, e)
            if key in seen:
                continue
            seen.add(key)
            slots.append({
                "date": iso_date,
                "startTime": s,
                "endTime": e,
                "assignToSelf": True
            })
        slots.sort(key=lambda x: x["startTime"])
        return slots


# -------------------------------------------------------------
# Совместимость со старым API
# -------------------------------------------------------------
class SlotParser:
    """Совместимый класс: сканирует папку с изображениями"""
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.inner = NewFormatSlotParser()

    def process_all_screenshots(self) -> List[dict]:
        all_slots = []
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}")
            return []
        image_files = [
            os.path.join(self.base_path, f)
            for f in os.listdir(self.base_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not image_files:
            logger.warning(f"No images in {self.base_path}")
            return []
        image_files.sort(key=lambda x: os.path.getctime(x))
        for fp in image_files:
            try:
                slots = self.inner.process_image(fp)
                all_slots.extend(slots)
            except Exception as e:
                logger.error(f"Error processing {fp}: {e}")
                continue
        unique = []
        seen = set()
        for s in all_slots:
            key = (s['date'], s['startTime'], s['endTime'])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        unique.sort(key=lambda x: (x['date'], x['startTime']))
        return unique


class MemorySlotParser(SlotParser):
    """Для обработки изображений из памяти (BytesIO)"""
    def __init__(self):
        super().__init__(base_path="")
        self.inner = NewFormatSlotParser()

    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[dict]:
        try:
            return self.inner.process_image(image_bytes)
        except Exception as e:
            logger.error(f"process_screenshot_from_memory error: {e}")
            return []


__all__ = ["SlotParser", "MemorySlotParser"]
