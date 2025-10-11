# bot/ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль (улучшенная версия).
- Надёжнее находит число дня (жёлтая подсветка или fallback по OCR на верхней полосе).
- Извлекает временные слоты формата HH:MM - HH:MM.
- Совместим с прежним API (SlotParser, MemorySlotParser).
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

logger = logging.getLogger("lavka.ocr_module")
logger.setLevel(logging.INFO)

import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

TIME_RANGE_RE = re.compile(r"(\d{1,2}[:.]\d{2})\s*[-–—]\s*(\d{1,2}[:.]\d{2})")


class NewFormatSlotParser:
    def __init__(self, now: Optional[datetime] = None, debug: bool = False, debug_dir: str = "/tmp/ocr_debug"):
        """
        debug: если True — сохранит вспомогательные артефакты и raw OCR text в debug_dir
        """
        self.now = now or datetime.now()
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    @staticmethod
    def _read_image(image_input) -> Optional[np.ndarray]:
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
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        scale = 1.4
        h, w = gray.shape[:2]
        new_w = max(100, int(w * scale))
        new_h = max(50, int(h * scale))
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
        """
        Поиск жёлтой подсветки в верхней полосе.
        Усилен: расширен H-диапазон, морфология.
        """
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            # Ослабленный диапазон жёлтого (покрывает более широкий спектр)
            lower = np.array([8, 60, 80])   # H небольшой, S и V помягче
            upper = np.array([50, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            # Ограничим область поиска верхней третью/четвертью изображения
            h = mask.shape[0]
            top_limit = int(h * 0.35)  # ищем в верхних 35%
            mask[ top_limit: , :] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:6]:
                area = cv2.contourArea(cnt)
                if area < 200:  # игнор мелких артефактов
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # Жёлтая подсветка должна быть в верхней части экрана
                if y > int(mask.shape[0] * 0.4):
                    continue
                return (x, y, w, h)
            return None
        except Exception as e:
            logger.error(f"_find_yellow_region error: {e}")
            return None

    @staticmethod
    def _ocr_day_number_from_roi(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        try:
            x, y, w, h = bbox
            pad_x = max(6, int(w * 0.16))
            pad_y = max(4, int(h * 0.16))
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

            config = "--psm 7 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(thr, config=config, lang='eng+rus')
            if not text:
                return None
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

    def _ocr_top_strip_choose_largest_digit(self, bgr: np.ndarray) -> Optional[int]:
        """
        Fallback: берем верхнюю полосу изображения, прогоняем image_to_data и выбираем
        числовой токен с наибольшей высотой (высота bbox => крупный шрифт => день).
        """
        try:
            h = bgr.shape[0]
            top_h = max(40, int(h * 0.25))
            top_strip = bgr[0:top_h, :].copy()
            proc = self._preprocess_for_ocr(top_strip)
            pil = Image.fromarray(proc)
            # получаем детализированную информацию
            data = pytesseract.image_to_data(pil, config="--psm 6", lang='eng+rus', output_type=pytesseract.Output.DICT)
            best = None
            best_h = 0
            for i, txt in enumerate(data.get("text", [])):
                if not txt or not re.search(r"\d", txt):
                    continue
                # вычисляем bbox height
                try:
                    hbox = int(data["height"][i])
                except:
                    hbox = 0
                # Также можно смотреть правую/левую координату, но берём просто максимальную высоту
                if hbox > best_h:
                    digits = re.findall(r"\d{1,2}", txt)
                    if not digits:
                        continue
                    val = int(digits[0])
                    if 1 <= val <= 31:
                        best_h = hbox
                        best = val
            if best:
                return best

            # если не нашли в top_strip — попробуем по всей картинке, но учитывать y (чтобы ближе к верху)
            pil_full = Image.fromarray(self._preprocess_for_ocr(bgr))
            data = pytesseract.image_to_data(pil_full, config="--psm 6", lang='eng+rus', output_type=pytesseract.Output.DICT)
            best = None
            best_score = -9999
            for i, txt in enumerate(data.get("text", [])):
                if not txt or not re.search(r"\d", txt):
                    continue
                try:
                    val = int(re.findall(r"\d{1,2}", txt)[0])
                except:
                    continue
                if not (1 <= val <= 31):
                    continue
                try:
                    y = int(data.get("top", [0])[i])
                    hbox = int(data.get("height", [0])[i])
                except:
                    y = 100000
                    hbox = 0
                # closer to top => higher score; larger height => higher score
                score = -y + (hbox * 2)
                if score > best_score:
                    best_score = score
                    best = val
            return best
        except Exception as e:
            logger.error(f"_ocr_top_strip_choose_largest_digit error: {e}")
            return None

    def _resolve_date_from_day(self, day: int) -> Optional[date]:
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
        try:
            proc = self._preprocess_for_ocr(bgr)
            pil = Image.fromarray(proc)
            config = "--psm 6 -c tessedit_char_whitelist=0123456789:.-–—"
            raw = pytesseract.image_to_string(pil, config=config, lang='eng+rus')
            if not raw or len(raw.strip()) == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pil2 = Image.fromarray(gray)
                raw = pytesseract.image_to_string(pil2, config=config, lang='eng+rus')

            if self.debug:
                fn = os.path.join(self.debug_dir, f"raw_ocr_{int(datetime.now().timestamp())}.txt")
                try:
                    with open(fn, "w", encoding="utf-8") as f:
                        f.write(raw or "")
                except:
                    pass

            matches = TIME_RANGE_RE.findall(raw or "")
            results = []
            for m in matches:
                t0 = m[0].replace(".", ":")
                t1 = m[1].replace(".", ":")
                def norm(t):
                    parts = t.split(":")
                    if len(parts) != 2:
                        return None
                    try:
                        hh = int(parts[0])
                        mm = int(parts[1])
                    except:
                        return None
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
        """
        Главная функция: возвращает список слотов в API-формате.
        """
        img = self._read_image(image_input)
        if img is None:
            logger.error("Cannot read image")
            return []

        # пробуем найти жёлтую область
        bbox = self._find_yellow_region(img)
        day = None
        if bbox:
            day = self._ocr_day_number_from_roi(img, bbox)

        if not day:
            # fallback: OCR в верхней полосе (и далее — по всей картинке)
            day = self._ocr_top_strip_choose_largest_digit(img)

        if not day:
            logger.error("Day number not detected")
            if self.debug:
                dumpfn = os.path.join(self.debug_dir, f"debug_img_{int(datetime.now().timestamp())}.png")
                try:
                    cv2.imwrite(dumpfn, img)
                    logger.info(f"Debug image saved: {dumpfn}")
                except:
                    pass
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
        logger.info(f"OCR result for date {iso_date}: {len(slots)} slot(s) found")
        return slots


# ---------- Совместимость со старым API ----------
class SlotParser:
    """Сканирует папку с изображениями — совместимый интерфейс."""
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.inner = NewFormatSlotParser(debug=debug)

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
    """Работает с BytesIO — добавлены алиасы для совместимости."""
    def __init__(self, debug: bool = False):
        super().__init__(base_path="", debug=debug)
        self.inner = NewFormatSlotParser(debug=debug)

    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[dict]:
        return self._call_inner(image_bytes)

    # алиасы (на всякий случай, чтобы избежать ошибок 'object has no attribute ...')
    def process_screenshot(self, image_bytes: BytesIO, is_last: bool = False) -> List[dict]:
        return self._call_inner(image_bytes)

    def process_image(self, image_input) -> List[dict]:
        return self._call_inner(image_input)

    def process_image_bytes(self, image_bytes: BytesIO) -> List[dict]:
        return self._call_inner(image_bytes)

    def _call_inner(self, image_input) -> List[dict]:
        try:
            return self.inner.process_image(image_input)
        except Exception as e:
            logger.error(f"Memory parser error: {e}")
            return []


__all__ = ["SlotParser", "MemorySlotParser"]
