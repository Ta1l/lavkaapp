# bot/ocr_module.py
# -*- coding: utf-8 -*-
"""
Улучшенный OCR-модуль для нового формата скриншотов.
Экспортирует: SlotParser, MemorySlotParser (совместимо со старым API).
Файлы отладки/сырой OCR сохраняются в /tmp/ocr_debug (включено по умолчанию).
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

# Путь к tesseract (на windows поправьте)
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

# Для image_to_data output
from pytesseract import Output

# Регэкспы
TIME_RANGE_RE = re.compile(r"(\d{1,2}[:.\s]\d{2})\s*[-–—]\s*(\d{1,2}[:.\s]\d{2})")
TIME_TOKEN_RE = re.compile(r"(\d{1,2}[:.\s]\d{2})")
DIGIT_RE = re.compile(r"\d{1,2}")

# Директория дебага
DEBUG_DIR = "/tmp/ocr_debug"


# ----------------- Вспомогательные функции -----------------
def _safe_read_image(image_input) -> Optional[np.ndarray]:
    """Возвращает BGR numpy array или None."""
    try:
        if isinstance(image_input, str):
            return cv2.imread(image_input)
        if isinstance(image_input, BytesIO):
            image_input.seek(0)
            pil = Image.open(image_input).convert("RGB")
            arr = np.array(pil)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"_safe_read_image error: {e}")
    return None


def _preprocess_gray_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """Универсальная предобработка — возвращает бинарное изображение."""
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


# ----------------- Основной парсер -----------------
class NewFormatSlotParser:
    def __init__(self, now: Optional[datetime] = None, debug: bool = True, debug_dir: str = DEBUG_DIR):
        """
        debug=True включит сохранение /tmp/ocr_debug/raw_ocr_{ts}.txt и debug images.
        (По умолчанию включено, чтобы быстрее отладить распознавание на реальных скриншотах.)
        """
        self.now = now or datetime.now()
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
            except Exception:
                pass

    # ---------- чтение и предобработка ----------
    def _read(self, image_input) -> Optional[np.ndarray]:
        return _safe_read_image(image_input)

    def _find_yellow_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Ищет жёлтую подсветку в верхней части экрана (возвращает bbox или None)."""
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            # расширенный жёлтый диапазон (чтобы покрыть разные оттенки)
            lower = np.array([8, 60, 80])
            upper = np.array([50, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

            # Ограничим верхнюю часть (обычно там календарная полоска)
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
            for cnt in contours[:6]:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if y > int(mask.shape[0] * 0.45):
                    continue
                return (x, y, w, h)
        except Exception as e:
            logger.error(f"_find_yellow_bbox error: {e}")
        return None

    def _ocr_day_from_bbox(self, bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        try:
            x, y, w, h = bbox
            pad_x = max(6, int(w * 0.15))
            pad_y = max(4, int(h * 0.15))
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
            txt = pytesseract.image_to_string(thr, config=config, lang='eng+rus')
            digits = DIGIT_RE.findall(txt or "")
            if digits:
                num = int(digits[0])
                if 1 <= num <= 31:
                    return num
        except Exception as e:
            logger.error(f"_ocr_day_from_bbox error: {e}")
        return None

    def _ocr_day_fallback_topstrip(self, bgr: np.ndarray) -> Optional[int]:
        """Fallback: находим наиболее вероятный цифровой токен в верхней полосе через image_to_data."""
        try:
            h = bgr.shape[0]
            top_h = max(40, int(h * 0.25))
            top_strip = bgr[:top_h, :].copy()
            proc = _preprocess_gray_for_ocr(top_strip)
            pil = Image.fromarray(proc)
            data = pytesseract.image_to_data(pil, output_type=Output.DICT, lang='eng+rus', config="--psm 6")
            best = None
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
                    best = val
            if best:
                return best

            # глобальный fallback по картинке, с приоритетом на верхние координаты
            pil_full = Image.fromarray(_preprocess_gray_for_ocr(bgr))
            data = pytesseract.image_to_data(pil_full, output_type=Output.DICT, lang='eng+rus', config="--psm 6")
            best = None
            best_score = -10**9
            for i, txt in enumerate(data.get("text", [])):
                if not txt or not DIGIT_RE.search(txt):
                    continue
                digits = DIGIT_RE.findall(txt)
                if not digits:
                    continue
                v = int(digits[0])
                if not (1 <= v <= 31):
                    continue
                try:
                    top = int(data.get("top", [0])[i])
                    hbox = int(data.get("height", [0])[i])
                except:
                    top = 100000
                    hbox = 0
                score = -top + hbox * 2
                if score > best_score:
                    best_score = score
                    best = v
            return best
        except Exception as e:
            logger.error(f"_ocr_day_fallback_topstrip error: {e}")
            return None

    # ---------- извлечение строк (coords) ----------
    def _extract_lines_with_coords(self, bgr: np.ndarray) -> List[dict]:
        """
        Возвращает список {"text": str, "y": int, "x": int} отсортированных по координатам.
        Использует pytesseract.image_to_data.
        """
        try:
            pil = Image.fromarray(_preprocess_gray_for_ocr(bgr))
            data = pytesseract.image_to_data(pil, output_type=Output.DICT, lang='eng+rus', config="--psm 6")
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
                parts = [ (data["text"][j] or "").strip() for j in idxs if (data["text"][j] or "").strip() ]
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

    # ---------- парсинг времени ----------
    def _parse_time_pair(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Ищет в тексте пару времен. Возвращает normalized ("HH:MM","HH:MM") или None.
        Поддерживает разные разделители: :, ., пробел.
        """
        # 1) попробуем расширенный паттерн
        m = TIME_RANGE_RE.search(text)
        if m:
            t0 = m.group(1).replace(".", ":").replace(" ", ":")
            t1 = m.group(2).replace(".", ":").replace(" ", ":")
            def check(t):
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
            n0 = check(t0); n1 = check(t1)
            if n0 and n1:
                return (n0, n1)

        # 2) если не найдено, попробуем найти два токена времени подряд
        tokens = TIME_TOKEN_RE.findall(text)
        if len(tokens) >= 2:
            t0 = tokens[0].replace(".", ":").replace(" ", ":")
            t1 = tokens[1].replace(".", ":").replace(" ", ":")
            def check2(t):
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
            n0 = check2(t0); n1 = check2(t1)
            if n0 and n1:
                return (n0, n1)
        return None

    def _find_time_by_combining_lines(self, lines: List[dict], idx: int, max_window: int = 3,
                                      band_top: int = -10**9, band_bottom: int = 10**9) -> Optional[Tuple[int, str, str]]:
        """
        Комбинирует соседние строки (макс max_window) и ищет пару времени.
        Возвращает (start_idx, start, end) или None.
        """
        n = len(lines)
        for w in range(1, max_window+1):
            j = idx + w - 1
            if j >= n:
                break
            if lines[j]["y"] < band_top or lines[j]["y"] > band_bottom:
                break
            combined = " ".join(lines[k]["text"] for k in range(idx, j+1))
            res = self._parse_time_pair(combined)
            if res:
                return (idx, res[0], res[1])
        # try previous + current
        if idx - 1 >= 0 and lines[idx-1]["y"] >= band_top and lines[idx-1]["y"] <= band_bottom:
            combined = lines[idx-1]["text"] + " " + lines[idx]["text"]
            res = self._parse_time_pair(combined)
            if res:
                return (idx-1, res[0], res[1])
        return None

    # ---------- главный процесс ----------
    def process_image(self, image_input) -> List[dict]:
        """
        На вход: path / BytesIO / PIL.Image
        Возвращает список слотов:
        [{"date":"YYYY-MM-DD","startTime":"HH:MM","endTime":"HH:MM","assignToSelf": True}, ...]
        """
        img = self._read(image_input)
        if img is None:
            logger.error("Cannot read image for OCR")
            return []

        # 1) распознаём день
        bbox = self._find_yellow_bbox(img)
        day = None
        if bbox:
            day = self._ocr_day_from_bbox(img, bbox)
        if not day:
            day = self._ocr_day_fallback_topstrip(img)
        if not day:
            logger.error("Day number not detected")
            if self.debug:
                try:
                    fn = os.path.join(self.debug_dir, f"dbg_img_{int(datetime.now().timestamp())}.png")
                    cv2.imwrite(fn, img)
                    logger.info(f"Saved debug image: {fn}")
                except:
                    pass
            return []

        # 2) разрешаем месяц/год
        resolved = self._resolve_date_from_day(day)
        if not resolved:
            logger.error("Could not resolve date from day number")
            return []
        iso_date = resolved.isoformat()

        # 3) извлекаем линии с координатами и ищем времени
        lines = self._extract_lines_with_coords(img)
        if not lines:
            logger.info("No OCR lines extracted")
        slots = []
        seen = set()

        # поскольку каждый скриншот — отдельный день, просто ищем все временные диапазоны по всему изображению
        for i in range(len(lines)):
            found = self._find_time_by_combining_lines(lines, i, max_window=3)
            if not found:
                continue
            start_idx, st, et = found
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

        # fallback: если не найдено через lines — делаем глобальную OCR строку и ищем там
        if not slots:
            proc = _preprocess_gray_for_ocr(img)
            pil = Image.fromarray(proc)
            conf = "--psm 6 -c tessedit_char_whitelist=0123456789:.-–—"
            raw = pytesseract.image_to_string(pil, config=conf, lang='eng+rus')
            if self.debug:
                try:
                    fn = os.path.join(self.debug_dir, f"raw_ocr_{int(datetime.now().timestamp())}.txt")
                    with open(fn, "w", encoding="utf-8") as f:
                        f.write(raw or "")
                    logger.info(f"Raw OCR saved: {fn}")
                except:
                    pass
            # ищем пары
            matches = TIME_RANGE_RE.findall(raw or "")
            for m in matches:
                t0 = m[0].replace(".", ":").replace(" ", ":")
                t1 = m[1].replace(".", ":").replace(" ", ":")
                def norm(t):
                    try:
                        hh, mm = t.split(":")
                        hh = int(hh); mm = int(mm)
                        if 0 <= hh < 24 and 0 <= mm < 60:
                            return f"{hh:02d}:{mm:02d}"
                    except:
                        return None
                    return None
                n0 = norm(t0); n1 = norm(t1)
                if n0 and n1:
                    key = (iso_date, n0, n1)
                    if key not in seen:
                        seen.add(key)
                        slots.append({
                            "date": iso_date,
                            "startTime": n0,
                            "endTime": n1,
                            "assignToSelf": True
                        })

        if not slots:
            logger.info("No time ranges found")
            return []

        # сортируем по времени начала
        slots.sort(key=lambda x: x["startTime"])
        logger.info(f"OCR result for {iso_date}: {len(slots)} slot(s) found")
        return slots

    def _resolve_date_from_day(self, day: int) -> Optional[date]:
        today = self.now.date()
        candidates = []
        year = today.year
        month = today.month

        def try_make(y, m, d):
            try:
                return date(y, m, d)
            except:
                return None

        candidates.append(try_make(year, month, day))
        for delta in (-1, 1, 2):
            m = month + delta
            y = year
            if m < 1:
                m += 12; y -= 1
            elif m > 12:
                m -= 12; y += 1
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
                    best = c; best_delta = delta
        if best is None:
            future = [(c, (c - today).days) for c in candidates if (c - today).days > 0]
            if future:
                return min(future, key=lambda x: x[1])[0]
            return min(candidates, key=lambda c: abs((c - today).days))
        return best


# ----------------- Совместимость (старый API) -----------------
class SlotParser:
    """Совместимый интерфейс: обходит папку и возвращает все слоты (каждый файл — день)."""
    def __init__(self, base_path: str, debug: bool = True):
        self.base_path = base_path
        self.inner = NewFormatSlotParser(debug=debug)

    def process_all_screenshots(self) -> List[dict]:
        all_slots = []
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}")
            return []
        files = [os.path.join(self.base_path, f) for f in os.listdir(self.base_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            logger.warning(f"No images in {self.base_path}")
            return []
        files.sort(key=lambda x: os.path.getctime(x))
        for fp in files:
            try:
                slots = self.inner.process_image(fp)
                all_slots.extend(slots)
            except Exception as e:
                logger.error(f"Error processing {fp}: {e}")
                continue
        # уникализируем
        uniq = []
        seen = set()
        for s in all_slots:
            key = (s['date'], s['startTime'], s['endTime'])
            if key not in seen:
                seen.add(key); uniq.append(s)
        uniq.sort(key=lambda x: (x['date'], x['startTime']))
        return uniq


class MemorySlotParser(SlotParser):
    """Парсер для BytesIO: добавлены алиасы для совместимости с разными вызовами."""
    def __init__(self, debug: bool = True):
        super().__init__(base_path="", debug=debug)
        self.inner = NewFormatSlotParser(debug=debug)

    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[dict]:
        return self._call_inner(image_bytes)

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
            logger.error(f"MemorySlotParser error: {e}")
            return []


__all__ = ["SlotParser", "MemorySlotParser"]
