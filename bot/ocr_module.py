# bot/ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для нового формата скриншотов.
Каждый скриншот = отдельный день.
Месяц/год фиксируем: October 2025 (2025-10-XX).
Экспорт: SlotParser, MemorySlotParser (совместимо со старым API).
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

# Регулярка для временных диапазонов формата "08:00 - 13:00" (поддерживает :, .)
TIME_RANGE_RE = re.compile(r"(\d{1,2}[:.]\d{2})\s*[-–—]\s*(\d{1,2}[:.]\d{2})")
TIME_TOKEN_RE = re.compile(r"(\d{1,2}[:.]\d{2})")
DIGIT_RE = re.compile(r"\d{1,2}")

# Фиксированный месяц/год
FIXED_YEAR = 2025
FIXED_MONTH = 10  # октябрь


# --------- Вспомогательные функции ---------
def _read_image(image_input: Any) -> Optional[np.ndarray]:
    """Принимает путь str, BytesIO, bytes, PIL.Image -> возвращает BGR numpy array."""
    try:
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
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
    except Exception as e:
        logger.error(f"_read_image error: {e}")
    return None


def _preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """Грейскейл + CLAHE + blur + OTSU => бинаризация, инверт при необходимости."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = 1.4
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


def _normalize_time_token(t: str) -> Optional[str]:
    """Нормализует '8:00' или '08.00' или '8 00' в '08:00' или None."""
    t = t.replace(".", ":").replace(" ", ":")
    parts = t.split(":")
    if len(parts) != 2:
        return None
    try:
        hh = int(parts[0]); mm = int(parts[1])
    except:
        return None
    if 0 <= hh < 24 and 0 <= mm < 60:
        return f"{hh:02d}:{mm:02d}"
    return None


# --------- Основной класс парсера для нового формата ---------
class NewFormatSlotParser:
    def __init__(self, debug: bool = False, debug_dir: str = "/tmp/ocr_debug"):
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
            except:
                pass

    def _find_yellow_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Поиск самой большой жёлтой области в верхней части изображения.
        Вернёт bbox (x, y, w, h) или None.
        """
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            # Довольно широкий диапазон жёлтого/оранжевого
            lower = np.array([8, 70, 60])
            upper = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            # Ограничение поиска верхней частью экрана (обычно там календарь)
            h = mask.shape[0]
            top_limit = int(h * 0.35)
            mask[top_limit:, :] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # гарантируем, что bbox в верхней части
                if y > int(mask.shape[0] * 0.45):
                    continue
                return (x, y, w, h)
        except Exception as e:
            logger.error(f"_find_yellow_bbox error: {e}")
        return None

    def _ocr_day_from_bbox(self, bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """OCR только цифр внутри bbox."""
        try:
            x, y, w, h = bbox
            pad_x = max(6, int(w * 0.12))
            pad_y = max(4, int(h * 0.12))
            x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
            x1 = min(bgr.shape[1], x + w + pad_x); y1 = min(bgr.shape[0], y + h + pad_y)
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
            digits = DIGIT_RE.findall(text)
            if not digits:
                return None
            val = int(digits[0])
            if 1 <= val <= 31:
                return val
        except Exception as e:
            logger.error(f"_ocr_day_from_bbox error: {e}")
        return None

    def _ocr_day_fallback_topstrip(self, bgr: np.ndarray) -> Optional[int]:
        """Fallback: OCR верхней полосы, выбираем самый большой по bbox height цифровой токен."""
        try:
            h = bgr.shape[0]
            top_h = max(40, int(h * 0.25))
            top_strip = bgr[:top_h, :].copy()
            proc = _preprocess_for_ocr(top_strip)
            pil = Image.fromarray(proc)
            data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, lang='eng+rus', config="--psm 6")
            best_val = None
            best_h = 0
            for i, txt in enumerate(data.get("text", [])):
                if not txt or not DIGIT_RE.search(txt):
                    continue
                try:
                    hbox = int(data.get("height", [0])[i])
                except:
                    hbox = 0
                digits = DIGIT_RE.findall(txt)
                if not digits:
                    continue
                val = int(digits[0])
                if not (1 <= val <= 31):
                    continue
                if hbox > best_h:
                    best_h = hbox
                    best_val = val
            if best_val:
                return best_val

            # глобальный fallback: по всей картинке с приоритетом на верхние позиции
            pil2 = Image.fromarray(_preprocess_for_ocr(bgr))
            data2 = pytesseract.image_to_data(pil2, output_type=pytesseract.Output.DICT, lang='eng+rus', config="--psm 6")
            best_score = -10**9
            best_val = None
            for i, txt in enumerate(data2.get("text", [])):
                if not txt or not DIGIT_RE.search(txt):
                    continue
                digits = DIGIT_RE.findall(txt)
                if not digits:
                    continue
                val = int(digits[0])
                if not (1 <= val <= 31):
                    continue
                try:
                    top = int(data2.get("top", [0])[i])
                    hbox = int(data2.get("height", [0])[i])
                except:
                    top = 100000; hbox = 0
                score = -top + hbox * 2
                if score > best_score:
                    best_score = score
                    best_val = val
            return best_val
        except Exception as e:
            logger.error(f"_ocr_day_fallback_topstrip error: {e}")
        return None

    def _extract_lines_with_coords(self, bgr: np.ndarray) -> List[Dict]:
        """Возвращает список {'text','y','x'} отсортированных по (y,x)."""
        try:
            proc = _preprocess_for_ocr(bgr)
            pil = Image.fromarray(proc)
            data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, lang='eng+rus', config="--psm 6")
            groups = {}
            n = len(data.get("text", []))
            for i in range(n):
                txt = (data["text"][i] or "").strip()
                if not txt:
                    continue
                key = (data.get("block_num", [0])[i], data.get("par_num", [0])[i], data.get("line_num", [0])[i])
                groups.setdefault(key, []).append(i)
            lines = []
            for key, idxs in groups.items():
                idxs.sort(key=lambda j: data.get("left", [0])[j])
                parts = [(data["text"][j] or "").strip() for j in idxs if (data["text"][j] or "").strip()]
                if not parts:
                    continue
                text = " ".join(parts)
                top = min(int(data.get("top", [0])[j]) for j in idxs)
                left = min(int(data.get("left", [0])[j]) for j in idxs)
                lines.append({"text": text, "y": int(top), "x": int(left)})
            lines.sort(key=lambda l: (l["y"], l["x"]))
            return lines
        except Exception as e:
            logger.error(f"_extract_lines_with_coords error: {e}")
            return []

    def _parse_time_in_text(self, text: str) -> Optional[Tuple[str, str]]:
        """Ищет пару времени в тексте и нормализует её."""
        m = TIME_RANGE_RE.search(text)
        if m:
            t0 = _normalize_time_token(m.group(1))
            t1 = _normalize_time_token(m.group(2))
            if t0 and t1:
                return t0, t1
        # fallback: два токена подряд
        tokens = TIME_TOKEN_RE.findall(text)
        if len(tokens) >= 2:
            t0 = _normalize_time_token(tokens[0])
            t1 = _normalize_time_token(tokens[1])
            if t0 and t1:
                return t0, t1
        return None

    def _find_time_by_combination(self, lines: List[Dict], idx: int, max_window: int = 3) -> Optional[Tuple[int, str, str]]:
        """Комбинируем соседние строки (1..max_window) чтобы поймать время."""
        n = len(lines)
        for w in range(1, max_window + 1):
            j = idx + w - 1
            if j >= n:
                break
            combined = " ".join(lines[k]["text"] for k in range(idx, j + 1))
            res = self._parse_time_in_text(combined)
            if res:
                return (idx, res[0], res[1])
        # try previous + current
        if idx - 1 >= 0:
            combined = lines[idx - 1]["text"] + " " + lines[idx]["text"]
            res = self._parse_time_in_text(combined)
            if res:
                return (idx - 1, res[0], res[1])
        return None

    def process_image(self, image_input: Any) -> List[Dict]:
        """
        На вход: path/BytesIO/bytes/PIL.Image.
        Возвращает список слотов в формате API:
        [{"date":"YYYY-MM-DD","startTime":"HH:MM","endTime":"HH:MM","assignToSelf": True}, ...]
        """
        img = _read_image(image_input)
        if img is None:
            logger.error("Cannot read image for OCR")
            return []

        # 1) найти желтую обводку и распознать число
        day = None
        bbox = self._find_yellow_bbox(img)
        if bbox:
            day = self._ocr_day_from_bbox(img, bbox)

        if day is None:
            day = self._ocr_day_fallback_topstrip(img)

        if day is None:
            logger.error("Day number not detected")
            if self.debug:
                try:
                    fn = os.path.join(self.debug_dir, f"dbg_img_{int(__import__('time').time())}.png")
                    cv2.imwrite(fn, img)
                    logger.info(f"Saved debug image to {fn}")
                except Exception:
                    pass
            return []

        # 2) Формируем дату: фиксируем октябрь 2025
        try:
            resolved_date = date(FIXED_YEAR, FIXED_MONTH, int(day))
            iso_date = resolved_date.isoformat()
        except Exception as e:
            logger.error(f"Invalid day extracted: {day} ({e})")
            return []

        # 3) Извлечение слотов по времени
        lines = self._extract_lines_with_coords(img)
        slots: List[Dict] = []
        seen = set()

        # Ищем с помощью комбинирования строк
        for i in range(len(lines)):
            found = self._find_time_by_combination(lines, i, max_window=3)
            if not found:
                continue
            _, st, et = found
            key = (iso_date, st, et)
            if key in seen:
                continue
            seen.add(key)
            slots.append({
                "date": iso_date,
                "startTime": st,
                "endTime": et,
                "assignToSelf": True
            })

        # fallback: если ничего не найдено — искать по всему тексту OCR
        if not slots:
            proc = _preprocess_for_ocr(img)
            pil = Image.fromarray(proc)
            config = "--psm 6 -c tessedit_char_whitelist=0123456789:.-–—"
            raw = pytesseract.image_to_string(pil, config=config, lang='eng+rus')
            if self.debug:
                try:
                    fn = os.path.join(self.debug_dir, f"raw_ocr_{int(__import__('time').time())}.txt")
                    with open(fn, "w", encoding="utf-8") as f:
                        f.write(raw or "")
                    logger.info(f"Saved raw OCR to {fn}")
                except Exception:
                    pass
            matches = TIME_RANGE_RE.findall(raw or "")
            for m in matches:
                t0 = _normalize_time_token(m[0])
                t1 = _normalize_time_token(m[1])
                if t0 and t1:
                    key = (iso_date, t0, t1)
                    if key not in seen:
                        seen.add(key)
                        slots.append({
                            "date": iso_date,
                            "startTime": t0,
                            "endTime": t1,
                            "assignToSelf": True
                        })

        # Сортировка по времени начала
        slots.sort(key=lambda s: s["startTime"])
        logger.info(f"OCR ({iso_date}): found {len(slots)} slot(s)")
        return slots


# --------- Совместимость со старым API (SlotParser / MemorySlotParser) ---------
class SlotParser:
    """
    Совместимая оболочка: принимает путь к папке со скриншотами, где каждый файл — день.
    Использует NewFormatSlotParser для логики.
    """
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.inner = NewFormatSlotParser(debug=debug)

    def process_all_screenshots(self) -> List[Dict]:
        all_slots: List[Dict] = []
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}")
            return []
        files = [os.path.join(self.base_path, f) for f in os.listdir(self.base_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            logger.warning(f"No image files found in: {self.base_path}")
            return []
        files.sort(key=lambda x: os.path.getctime(x))
        for fp in files:
            try:
                slots = self.inner.process_image(fp)
                all_slots.extend(slots)
            except Exception as e:
                logger.error(f"Error processing screenshot {fp}: {e}")
                continue
        # Уникализируем
        uniq = []
        seen = set()
        for s in all_slots:
            key = (s["date"], s["startTime"], s["endTime"])
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        uniq.sort(key=lambda s: (s["date"], s["startTime"]))
        return uniq


class MemorySlotParser(SlotParser):
    """
    Для обработки BytesIO/bytes/PIL.Image.
    Совместимые методы:
      - process_screenshot_from_memory(BytesIO, is_last=False)
      - process_screenshot(BytesIO, is_last=False)
      - process_image(BytesIO)
      - process_image_bytes(bytes)
    """
    def __init__(self, debug: bool = False):
        super().__init__(base_path="", debug=debug)
        self.inner = NewFormatSlotParser(debug=debug)

    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        return self._call_inner(image_bytes)

    def process_screenshot(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        return self._call_inner(image_bytes)

    def process_image(self, image_input: Any) -> List[Dict]:
        return self._call_inner(image_input)

    def process_image_bytes(self, image_bytes: bytes) -> List[Dict]:
        return self._call_inner(image_bytes)

    def _call_inner(self, image_input: Any) -> List[Dict]:
        try:
            return self.inner.process_image(image_input)
        except Exception as e:
            logger.error(f"MemorySlotParser error: {e}")
            return []


# Явный экспорт
__all__ = ["SlotParser", "MemorySlotParser"]
