# bot/new_format_ocr.py
# -*- coding: utf-8 -*-
"""
Парсер для нового формата скриншота (каждый скриншот = один день).
- Находит желтую обводку сверху и извлекает номер дня внутри неё (цифра).
- Разрешает месяц/год исходя из текущей даты (пользователь присылает максимум следующую неделю).
- Извлекает все временные слоты формата "08:00 - 13:00".
- Возвращает список слотов в формате API.
"""

from datetime import datetime, timedelta, date
from io import BytesIO
from typing import List, Tuple, Optional
import re
import logging
import cv2
import numpy as np
from PIL import Image
import pytesseract
import os

logger = logging.getLogger("lavka.new_format_ocr")
logger.setLevel(logging.INFO)

# Настройка tesseract (если нужно — изменять по ОС)
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

TIME_RANGE_RE = re.compile(r"(\d{1,2}[:.]\d{2})\s*[-–—]\s*(\d{1,2}[:.]\d{2})")
TIME_TOKEN_RE = re.compile(r"(\d{1,2}[:.]\d{2})")


class NewFormatSlotParser:
    def __init__(self, now: Optional[datetime] = None):
        """
        now: можно передать фиктивную 'сегодняшнюю' дату для тестов; по умолчанию — datetime.now()
        """
        self.now = now or datetime.now()

    @staticmethod
    def _read_image(image_input) -> Optional[np.ndarray]:
        """
        image_input: path str или BytesIO или PIL.Image
        Возвращает BGR numpy array или None.
        """
        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                return img
            if isinstance(image_input, BytesIO):
                image_input.seek(0)
                pil = Image.open(image_input).convert("RGB")
                arr = np.array(pil)
                # PIL -> RGB, convert to BGR for OpenCV
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
        """Универсальная предобработка для повышения качества OCR."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # увеличим
        scale = 1.4
        h, w = gray.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Если фон темный — инвертируем
        if np.mean(bw) < 127:
            bw = cv2.bitwise_not(bw)
        return bw

    @staticmethod
    def _find_yellow_region(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Ищет жёлтую область (обводку/подложку в календарной полосе).
        Возвращает bbox (x, y, w, h) или None.
        Работает по HSV-диапазону для жёлтого.
        """
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            # Диапазон жёлтого — можно подправить если будет отличаться у пользователей
            lower = np.array([12, 100, 100])  # H от ~12
            upper = np.array([40, 255, 255])  # H до ~40
            mask = cv2.inRange(hsv, lower, upper)

            # Уберём шум
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Выбираем самый большой контур (ожидаем, что желтая подсветка одна и заметная)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:5]:
                area = cv2.contourArea(cnt)
                if area < 100:  # игнор мелких
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # Ограничим по соотношению: жёлтая обводка обычно широкая и невысокая,
                # но чтобы быть гибкими — не проверяем сильно строго.
                return (x, y, w, h)
            return None
        except Exception as e:
            logger.error(f"_find_yellow_region error: {e}")
            return None

    @staticmethod
    def _ocr_day_number_from_roi(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """
        Извлекает цифру дня из заданной bbox области.
        Возвращает int день (1..31) или None.
        """
        try:
            x, y, w, h = bbox
            # Расширим bbox немного, чтобы захватить саму цифру внутри
            pad_x = max(4, int(w * 0.12))
            pad_y = max(2, int(h * 0.12))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(bgr.shape[1], x + w + pad_x)
            y1 = min(bgr.shape[0], y + h + pad_y)
            roi = bgr[y0:y1, x0:x1].copy()

            # Можем сначала искать ярко-жёлтую внутреннюю заливку (если цифра на белом фоне внутри)
            # но проще — конвертим в серое и усиливаем контраст
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Подберём порог адаптивно
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Немного морфологии для читаемости
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
            # OCR: только цифры
            config = "--psm 6 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thr, config=config, lang='eng+rus')
            if not text:
                return None
            # Оставим только цифры
            digits = re.findall(r"\d{1,2}", text)
            if not digits:
                return None
            # берем первое число (обычно внутри один)
            day = int(digits[0])
            if 1 <= day <= 31:
                return day
            return None
        except Exception as e:
            logger.error(f"_ocr_day_number_from_roi error: {e}")
            return None

    def _resolve_date_from_day(self, day: int) -> Optional[date]:
        """
        Разрешает месяц и год для числа day (1..31) исходя из текущей даты self.now.
        Алгоритм:
         - Собираем кандидатов: day в текущем месяце, в следующем месяце, в предыдущем месяце.
         - Выбираем кандидат, который находится в диапазоне [0 .. lookahead_days] от today (предпочтительно ближайший).
         - lookahead_days = 13 (покрывает "следующую неделю" и возможные смещения).
         - Если нет кандидатов в будущем — берем ближайший будущий, либо ближайший вообще.
        """
        today = self.now.date()
        candidates = []
        year = today.year
        month = today.month

        # helper to create date if valid
        def try_make(y, m, d):
            try:
                return date(y, m, d)
            except Exception:
                return None

        # current month
        candidates.append(try_make(year, month, day))
        # next month
        m = month + 1
        y = year
        if m == 13:
            m = 1
            y += 1
        candidates.append(try_make(y, m, day))
        # prev month
        m = month - 1
        y = year
        if m == 0:
            m = 12
            y -= 1
        candidates.append(try_make(y, m, day))
        # also next-next month (edge cases near month ends)
        m = month + 2
        y = year
        if m > 12:
            m -= 12
            y += 1
        candidates.append(try_make(y, m, day))

        # Filter None and deduplicate
        candidates = [c for c in candidates if c is not None]
        candidates = sorted(set(candidates))

        if not candidates:
            return None

        lookahead_days = 13  # покрывает неделю вперед и переходы
        best = None
        best_delta = 10**9

        for c in candidates:
            delta = (c - today).days
            # prefer non-negative deltas up to lookahead_days
            if 0 <= delta <= lookahead_days:
                if delta < best_delta:
                    best = c
                    best_delta = delta

        # если не нашли в 0..lookahead_days — возьмём ближайший будущий
        if best is None:
            future_candidates = [(c, (c - today).days) for c in candidates if (c - today).days > 0]
            if future_candidates:
                # минимальный положительный delta
                cmin = min(future_candidates, key=lambda x: x[1])[0]
                return cmin
            # иначе — ближайший вообще (может быть прошедший)
            cmin = min(candidates, key=lambda c: abs((c - today).days))
            return cmin

        return best

    def _extract_time_ranges(self, bgr: np.ndarray) -> List[Tuple[str, str]]:
        """
        Извлекает все вхождения 'HH:MM - HH:MM' из изображения.
        Возвращает список уникальных (start, end) в формате 'HH:MM'.
        """
        try:
            proc = self._preprocess_for_ocr(bgr)
            # Для OCR лучше использовать PIL Image
            pil = Image.fromarray(proc)
            # Конфиг: разрешаем цифры, двоеточие, дефис
            config = "--psm 6 -c tessedit_char_whitelist=0123456789:.-–—"
            raw = pytesseract.image_to_string(pil, config=config, lang='eng+rus')
            if not raw:
                # fallback: run on grayscale without threshold
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pil2 = Image.fromarray(gray)
                raw = pytesseract.image_to_string(pil2, config=config, lang='eng+rus')

            matches = TIME_RANGE_RE.findall(raw)
            results = []
            for m in matches:
                t0 = m[0].replace(".", ":")
                t1 = m[1].replace(".", ":")
                # Нормализуем время до HH:MM (добавляем ведущие нули если нужно)
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
            # Уникализируем и отсортируем по времени начала
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
        """
        Главная функция: на вход — путь к файлу или BytesIO или PIL.Image.
        Возвращает слоты в формате API:
        [
          {"date": "YYYY-MM-DD", "startTime": "HH:MM", "endTime":"HH:MM", "assignToSelf": True},
          ...
        ]
        """
        img = self._read_image(image_input)
        if img is None:
            logger.error("cannot read image")
            return []

        # 1) Найдём жёлтую область и распознаем число дня
        bbox = self._find_yellow_region(img)
        day = None
        if bbox:
            day = self._ocr_day_number_from_roi(img, bbox)
        else:
            logger.info("Yellow region not found — попробуем OCR по верхней полосе")
            # fallback: берем верхнюю часть изображения (верхние 20% высоты), ищем цифры там
            h = img.shape[0]
            top_strip = img[0:max(1, int(h * 0.22)), :]
            # run OCR
            try:
                pil = Image.fromarray(self._preprocess_for_ocr(top_strip))
                config = "--psm 6 -c tessedit_char_whitelist=0123456789"
                txt = pytesseract.image_to_string(pil, config=config, lang='eng+rus')
                digits = re.findall(r"\d{1,2}", txt)
                if digits:
                    day = int(digits[0])
            except Exception as e:
                logger.error(f"fallback top_strip OCR error: {e}")

        if not day:
            logger.error("Day number not detected")
            return []

        # 2) Разрешаем дату (месяц/год)
        resolved = self._resolve_date_from_day(day)
        if not resolved:
            logger.error("Could not resolve date from day number")
            return []

        iso_date = resolved.isoformat()

        # 3) Извлекаем временные слоты
        time_ranges = self._extract_time_ranges(img)
        if not time_ranges:
            logger.info("No time ranges found")
            return []

        # 4) Формируем результат
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

        # сортируем по времени начала
        slots.sort(key=lambda x: x["startTime"])
        return slots


# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = NewFormatSlotParser()
    # Подставьте ваш файл:
    test_path = "./test_screenshots/new_format_example.png"
    if not os.path.exists(test_path):
        logger.info(f"Place example screenshot to {test_path} to test parser.")
    else:
        res = parser.process_image(test_path)
        import json
        print(json.dumps(res, ensure_ascii=False, indent=2))