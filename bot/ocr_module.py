# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для нового формата скриншотов.
Каждый скриншот = отдельный день.
Месяц/год фиксируем: October 2025 (2025-10-XX).
Экспорт: SlotParser, MemorySlotParser, NewFormatSlotParser.
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
# Фиксированный месяц/год
FIXED_YEAR = 2025
FIXED_MONTH = 10  # октябрь

# Строгий паттерн временного диапазона HH:MM - HH:MM (поддержка разных тире)
STRICT_TIME_RANGE_RE = re.compile(
    r"\b([01]?\d|2[0-3]):([0-5]\d)\s*[-–—]\s*([01]?\d|2[0-3]):([0-5]\d)\b"
)

# Регулярка для извлечения чисел
DIGIT_RE = re.compile(r"\d{1,2}")


# --------- Вспомогательные функции ---------
def _read_image(image_input: Any) -> Optional[np.ndarray]:
    """Принимает путь str, BytesIO, bytes, PIL.Image -> возвращает BGR numpy array."""
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.error(f"Image file not found: {image_input}")
                return None
            img = cv2.imread(image_input)
            if img is None:
                logger.error(f"Failed to read image with cv2: {image_input}")
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
        logger.error(f"Unknown image input type: {type(image_input)}")
    except Exception as e:
        logger.error(f"_read_image error: {e}", exc_info=True)
    return None


def _resize_for_ocr(gray: np.ndarray, scale: float = 1.6) -> np.ndarray:
    """Увеличивает изображение для OCR, сохраняя соотношение сторон."""
    h, w = gray.shape[:2]
    new_w = max(100, int(w * scale))
    new_h = max(60, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """Грейскейл + CLAHE + blur + OTSU => бинаризация, инверт при необходимости."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _resize_for_ocr(gray, scale=1.4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)
    return bw


def _preprocess_strong_binary(bgr: np.ndarray) -> np.ndarray:
    """Более агрессивная предобработка: усиление контраста и жёсткая бинаризация."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _resize_for_ocr(gray, scale=1.8)
    gray = cv2.equalizeHist(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 11)
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw


def _normalize_dashes(text: str) -> str:
    """Унифицирует разные тире в обычный дефис '-'."""
    if not text:
        return ""
    for ch in ["–", "—", "−", "‐", "‑", "‒", "―"]:
        text = text.replace(ch, "-")
    return text


def _normalize_ocr_text_numbers(text: str) -> str:
    """Меняет буквы на цифры там, где это уместно."""
    if not text:
        return ""
    reps = {
        "O": "0", "o": "0", "Q": "0",
        "l": "1", "I": "1", "|": "1", "í": "1",
        "S": "5",
        "B": "8",
    }
    out = []
    for ch in text:
        out.append(reps.get(ch, ch))
    return "".join(out)


def _text_to_time_ranges(text: str) -> List[Tuple[str, str]]:
    """Парсит из текста все временные диапазоны 'HH:MM - HH:MM'."""
    if not text:
        return []
    # Нормализация похожих символов и тире
    text = _normalize_ocr_text_numbers(text)
    text = text.replace("：", ":").replace("∶", ":").replace("․", ":").replace("·", ":").replace("•", ":")
    text = text.replace(".", ":")
    text = _normalize_dashes(text)

    ranges: List[Tuple[str, str]] = []
    for m in STRICT_TIME_RANGE_RE.finditer(text):
        h1, m1, h2, m2 = m.groups()
        t0 = f"{int(h1):02d}:{int(m1):02d}"
        t1 = f"{int(h2):02d}:{int(m2):02d}"
        ranges.append((t0, t1))

    # Уникализация по порядку появления
    uniq = []
    seen = set()
    for t0, t1 in ranges:
        key = (t0, t1)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq


# --------- Основной класс парсера ---------
class NewFormatSlotParser:
    """
    Логика:
      1) Находим жёлтый блок наверху (в нём число дня), извлекаем день (1..31).
      2) Формируем дату: 2025-10-<day>.
      3) Ищем временные слоты строго 'HH:MM - HH:MM' в области ниже жёлтого блока.
    """
    def __init__(self, debug: bool = False, debug_dir: str = "/tmp/ocr_debug"):
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            try:
                if platform.system() == "Windows":
                    self.debug_dir = os.path.join(os.environ.get('TEMP', 'C:/Temp'), 'ocr_debug')
                os.makedirs(self.debug_dir, exist_ok=True)
                logger.info(f"Debug directory: {self.debug_dir}")
            except Exception as e:
                logger.error(f"Failed to create debug directory: {e}")

    # ---------- Поиск жёлтого блока ----------
    def _yellow_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Объединенная маска жёлтого в HSV и Lab."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Диапазоны желтого/оранжевого
        lower1 = np.array([12, 80, 80])
        upper1 = np.array([45, 255, 255])
        lower2 = np.array([8, 60, 60])
        upper2 = np.array([55, 255, 255])
        mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                                  cv2.inRange(hsv, lower2, upper2))

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        # Жёлтый в Lab = высокий b
        lower_lab = np.array([150, 100, 140])  # L, a, b
        upper_lab = np.array([255, 170, 205])
        mask_lab = cv2.inRange(lab, lower_lab, upper_lab)

        mask = cv2.bitwise_or(mask_hsv, mask_lab)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _find_yellow_bbox(self, bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Поиск самой крупной жёлтой области в верхней части изображения.
        Вернёт bbox (x, y, w, h) или None.
        """
        try:
            mask = self._yellow_mask(bgr)
            H, W = mask.shape[:2]
            # Ищем только в верхней части
            top_limit = int(H * 0.45)
            mask[top_limit:, :] = 0

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            best = None
            best_score = -1
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if y > int(H * 0.45):
                    continue
                if h < 16 or w < 16:
                    continue
                aspect = w / float(h + 1e-3)
                # Небольшая эвристика: берём большие и ближе к верху
                score = area - y * 5 + min(aspect, 3.0) * 10
                if score > best_score:
                    best_score = score
                    best = (x, y, w, h)

            if best and self.debug:
                try:
                    dbg = bgr.copy()
                    x, y, w, h = best
                    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.imwrite(os.path.join(self.debug_dir, "yellow_bbox_debug.png"), dbg)
                    cv2.imwrite(os.path.join(self.debug_dir, "yellow_mask.png"), mask)
                except Exception:
                    pass

            return best
        except Exception as e:
            logger.error(f"_find_yellow_bbox error: {e}", exc_info=True)
        return None

    # ---------- Извлечение дня из жёлтого блока (ИСПРАВЛЕНО) ----------
    def _ocr_day_from_bbox(self, bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[int]:
        """Распознать день (1..31) внутри жёлтого блока с несколькими попытками."""
        try:
            x, y, w, h = bbox
            pad_x = max(6, int(w * 0.15))
            pad_y = max(6, int(h * 0.15))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(bgr.shape[1], x + w + pad_x)
            y1 = min(bgr.shape[0], y + h + pad_y)
            
            # Проверяем, что область валидна
            if x1 <= x0 or y1 <= y0:
                logger.error(f"Invalid ROI bounds: ({x0},{y0}) to ({x1},{y1})")
                return None
                
            roi = bgr[y0:y1, x0:x1].copy()
            
            if roi.size == 0:
                logger.error("Empty ROI extracted")
                return None

            candidates: List[Tuple[int, int]] = []  # (value, confidence_like)

            # Попытка 1: Простая обработка ROI
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = _resize_for_ocr(gray, scale=2.0)
                _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Убираем жёлтый фон другим способом - инверсией при необходимости
                if np.mean(thr) < 127:
                    thr = cv2.bitwise_not(thr)
                    
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

                if self.debug:
                    try:
                        cv2.imwrite(os.path.join(self.debug_dir, "day_roi.png"), roi)
                        cv2.imwrite(os.path.join(self.debug_dir, "day_thr.png"), thr)
                    except Exception:
                        pass

                # PSM 8 для одного слова
                config = "--psm 8 -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(thr, config=config, lang='eng')
                digits = DIGIT_RE.findall(text or "")
                if digits:
                    val = int(digits[0])
                    if 1 <= val <= 31:
                        candidates.append((val, 85))
            except Exception as e:
                logger.debug(f"OCR attempt 1 failed: {e}")

            # Попытка 2: Данные с уверенностью
            try:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = _resize_for_ocr(gray, scale=2.0)
                _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                data = pytesseract.image_to_data(thr, output_type=pytesseract.Output.DICT, lang='eng',
                                                 config="--psm 6 -c tessedit_char_whitelist=0123456789")
                best_val = None
                best_score = -1
                n = len(data.get("text", []))
                for i in range(n):
                    txt = (data["text"][i] or "").strip()
                    if not txt or not DIGIT_RE.fullmatch(txt):
                        continue
                    try:
                        val = int(txt)
                        if not (1 <= val <= 31):
                            continue
                        conf = int(float(data.get("conf", [0])[i]))
                        hbox = int(data.get("height", [0])[i])
                    except Exception:
                        continue
                    score = conf + hbox * 2
                    if score > best_score:
                        best_score = score
                        best_val = val
                if best_val is not None:
                    candidates.append((best_val, min(best_score, 99)))
            except Exception as e:
                logger.debug(f"OCR attempt 2 failed: {e}")

            # Попытка 3: Альтернативная обработка
            try:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_gray = _resize_for_ocr(roi_gray, 2.0)
                
                # Применяем адаптивный порог вместо OTSU
                roi_thr = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                
                config = "--psm 7 -c tessedit_char_whitelist=0123456789"
                txt = pytesseract.image_to_string(roi_thr, config=config, lang='eng')
                digits = DIGIT_RE.findall(txt or "")
                if digits:
                    val = int(digits[0])
                    if 1 <= val <= 31:
                        candidates.append((val, 70))
            except Exception as e:
                logger.debug(f"OCR attempt 3 failed: {e}")

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
                
        except Exception as e:
            logger.error(f"_ocr_day_from_bbox error: {e}", exc_info=True)
        return None

    def _ocr_day_fallback_topstrip(self, bgr: np.ndarray) -> Optional[int]:
        """Fallback: сканируем верхнюю полосу и ищем цифру с максимальным покрытием жёлтым."""
        try:
            H = bgr.shape[0]
            top_h = max(50, int(H * 0.28))
            top_strip = bgr[:top_h, :].copy()

            # Маска жёлтого для приоритизации цифр внутри жёлтого
            ymask = self._yellow_mask(top_strip)

            proc = _preprocess_for_ocr(top_strip)
            data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, lang='eng',
                                             config="--psm 6 -c tessedit_char_whitelist=0123456789")
            best_val = None
            best_score = -1
            n = len(data.get("text", []))
            for i in range(n):
                txt = (data["text"][i] or "").strip()
                if not txt or not DIGIT_RE.fullmatch(txt):
                    continue
                try:
                    val = int(txt)
                    if not (1 <= val <= 31):
                        continue
                    x = int(float(data.get("left", [0])[i]))
                    y = int(float(data.get("top", [0])[i]))
                    w = int(float(data.get("width", [0])[i]))
                    h = int(float(data.get("height", [0])[i]))
                    conf = int(float(data.get("conf", [0])[i]))
                except Exception:
                    continue

                # Доля жёлтого под этим боксом
                x0, y0 = max(0, x - 3), max(0, y - 3)
                x1, y1 = min(ymask.shape[1], x + w + 3), min(ymask.shape[0], y + h + 3)
                
                # Проверяем корректность границ
                if x1 > x0 and y1 > y0:
                    box = ymask[y0:y1, x0:x1]
                    if box.size > 0:
                        yellow_ratio = float(np.count_nonzero(box)) / float(box.size)
                        score = conf + h * 2 + yellow_ratio * 200  # бонус за жёлтый фон
                        if score > best_score:
                            best_score = score
                            best_val = val

            if best_val:
                return best_val

            # Глобальный fallback: по всей картинке с приоритетом верх/высота
            proc2 = _preprocess_for_ocr(bgr)
            data2 = pytesseract.image_to_data(proc2, output_type=pytesseract.Output.DICT, lang='eng',
                                              config="--psm 6 -c tessedit_char_whitelist=0123456789")
            best_val = None
            best_score = -10**9
            n = len(data2.get("text", []))
            for i in range(n):
                txt = (data2["text"][i] or "").strip()
                if not txt or not DIGIT_RE.fullmatch(txt):
                    continue
                try:
                    val = int(txt)
                    top = int(data2.get("top", [0])[i])
                    hbox = int(data2.get("height", [0])[i])
                    conf = int(float(data2.get("conf", [0])[i]))
                except Exception:
                    continue
                if not (1 <= val <= 31):
                    continue
                score = -top * 2 + hbox * 3 + conf
                if score > best_score:
                    best_score = score
                    best_val = val
            return best_val
        except Exception as e:
            logger.error(f"_ocr_day_fallback_topstrip error: {e}", exc_info=True)
        return None

    # ---------- Вспомогательное: выбор ROI для поиска слотов ----------
    def _crop_time_region(self, bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Возвращает область для поиска времен: ниже жёлтого блока.
        Если bbox нет — отсекаем верхнюю шапку (примерно 20% высоты).
        """
        H, W = bgr.shape[:2]
        if bbox:
            x, y, w, h = bbox
            margin = max(6, int(H * 0.02))
            y_start = min(H, y + h + margin)
        else:
            y_start = int(H * 0.2)  # отрубить статус-бар/календарь
            
        # Проверяем корректность границ
        if y_start >= H:
            y_start = int(H * 0.2)
            
        roi = bgr[y_start:, :].copy()
        
        if self.debug:
            try:
                cv2.imwrite(os.path.join(self.debug_dir, "time_search_roi.png"), roi)
            except Exception:
                pass
        return roi

    # ---------- Извлечение строк с координатами ----------
    def _extract_lines_with_coords(self, bgr: np.ndarray, whitelist_digits_only: bool = False) -> List[Dict]:
        """Собирает строки через image_to_data с координатами, сортирует по (y,x)."""
        try:
            proc = _preprocess_for_ocr(bgr)
            pil = Image.fromarray(proc)
            lang = 'eng' if whitelist_digits_only else 'eng+rus'
            config = "--psm 6"
            if whitelist_digits_only:
                config += " -c tessedit_char_whitelist=0123456789:.-–— "
            data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, lang=lang, config=config)
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
                top = min(int(float(data.get("top", [0])[j])) for j in idxs)
                left = min(int(float(data.get("left", [0])[j])) for j in idxs)
                lines.append({"text": text, "y": int(top), "x": int(left)})
            lines.sort(key=lambda l: (l["y"], l["x"]))
            return lines
        except Exception as e:
            logger.error(f"_extract_lines_with_coords error: {e}", exc_info=True)
            return []

    # ---------- Поиск временных диапазонов ----------
    def _parse_time_in_text(self, text: str) -> List[Tuple[str, str]]:
        """Возвращает все пары времени из текста (строго HH:MM - HH:MM)."""
        return _text_to_time_ranges(text)

    def _find_time_ranges_in_roi(self, roi_bgr: np.ndarray) -> List[Tuple[str, str]]:
        """
        Ищет временные диапазоны в ROI несколькими способами и возвращает их в порядке появления.
        """
        found: List[Tuple[str, str]] = []

        # 1) Строки с whitelist цифр и разделителей
        lines_digits_only = self._extract_lines_with_coords(roi_bgr, whitelist_digits_only=True)
        for ln in lines_digits_only:
            items = self._parse_time_in_text(ln["text"])
            found.extend(items)

        # 2) Если ничего — обычные строки (на случай, если whitelist что-то съел)
        if not found:
            lines_mixed = self._extract_lines_with_coords(roi_bgr, whitelist_digits_only=False)
            for ln in lines_mixed:
                items = self._parse_time_in_text(ln["text"])
                found.extend(items)

        # 3) Сырой OCR на сильной бинаризации, whitelist
        if not found:
            proc = _preprocess_strong_binary(roi_bgr)
            pil = Image.fromarray(proc)
            config = "--psm 6 -c tessedit_char_whitelist=0123456789:.-–—"
            raw = pytesseract.image_to_string(pil, config=config, lang='eng')
            items = self._parse_time_in_text(raw or "")
            found.extend(items)
            if self.debug:
                try:
                    with open(os.path.join(self.debug_dir, "raw_ocr_times.txt"), "w", encoding="utf-8") as f:
                        f.write(raw or "")
                except Exception:
                    pass

        # Уникализация в порядке появления
        uniq = []
        seen = set()
        for t0, t1 in found:
            key = (t0, t1)
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        return uniq

    # ---------- Основной публичный метод ----------
    def process_image(self, image_input: Any) -> List[Dict]:
        """
        На вход: path/BytesIO/bytes/PIL.Image.
        Возвращает список слотов формата:
        [{"date":"YYYY-MM-DD","startTime":"HH:MM","endTime":"HH:MM","assignToSelf": True}, ...]
        """
        logger.info(f"Processing image, input type: {type(image_input)}")

        img = _read_image(image_input)
        if img is None:
            logger.error("Cannot read image for OCR")
            return []

        logger.info(f"Image loaded successfully, shape: {img.shape}")

        # 1) Найти жёлтый блок и распознать день
        day = None
        bbox = self._find_yellow_bbox(img)
        if bbox:
            logger.info(f"Yellow bbox found: {bbox}")
            day = self._ocr_day_from_bbox(img, bbox)
            logger.info(f"Day from bbox OCR: {day}")
        else:
            logger.warning("No yellow bbox found")

        if day is None:
            logger.info("Trying fallback day detection...")
            day = self._ocr_day_fallback_topstrip(img)
            logger.info(f"Day from fallback: {day}")

        if day is None:
            logger.error("Day number not detected")
            if self.debug:
                try:
                    import time
                    fn = os.path.join(self.debug_dir, f"dbg_img_{int(time.time())}.png")
                    cv2.imwrite(fn, img)
                    logger.info(f"Saved debug image to {fn}")
                except Exception as e:
                    logger.error(f"Failed to save debug image: {e}")
            return []

        # 2) Формируем дату: фиксируем октябрь 2025
        try:
            resolved_date = date(FIXED_YEAR, FIXED_MONTH, int(day))
            iso_date = resolved_date.isoformat()
            logger.info(f"Resolved date: {iso_date}")
        except Exception as e:
            logger.error(f"Invalid day extracted: {day} ({e})")
            return []

        # 3) Ищем временные слоты строго по формату HH:MM - HH:MM
        time_roi = self._crop_time_region(img, bbox)
        ranges = self._find_time_ranges_in_roi(time_roi)
        logger.info(f"Found {len(ranges)} time range(s)")

        slots: List[Dict] = []
        seen = set()
        for st, et in ranges:
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
            logger.info(f"Slot: {st} - {et}")

        if not slots:
            logger.warning("No time ranges found")

        # Важно: не сортируем — сохраняем порядок обнаружения (первый найденный — первый слот)
        logger.info(f"OCR result for date {iso_date}: found {len(slots)} slot(s)")
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
                logger.error(f"Error processing screenshot {fp}: {e}", exc_info=True)
                continue
        # Уникализируем
        uniq = []
        seen = set()
        for s in all_slots:
            key = (s["date"], s["startTime"], s["endTime"])
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        # Порядок по дате+времени для консистентности батча (без влияния на единичные вызовы)
        uniq.sort(key=lambda s: (s["date"], s["startTime"], s["endTime"]))
        return uniq


class MemorySlotParser:
    """
    Для обработки BytesIO/bytes/PIL.Image.
    Полностью совместимый класс со всеми возможными методами.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.inner = NewFormatSlotParser(debug=debug)
        self.base_path = ""
        self.accumulated_slots: List[Dict] = []

    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """Обрабатывает скриншот из памяти"""
        slots = self._call_inner(image_bytes)
        self.accumulated_slots.extend(slots)
        if is_last:
            return self.get_all_slots()
        return slots

    def process_screenshot(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """Алиас для process_screenshot_from_memory"""
        return self.process_screenshot_from_memory(image_bytes, is_last)

    def process_image(self, image_input: Any) -> List[Dict]:
        """Обрабатывает изображение любого типа"""
        return self._call_inner(image_input)

    def process_image_bytes(self, image_bytes: bytes) -> List[Dict]:
        """Обрабатывает изображение из байтов"""
        return self._call_inner(image_bytes)

    def get_all_slots(self) -> List[Dict]:
        """Возвращает все накопленные слоты"""
        uniq = []
        seen = set()
        for s in self.accumulated_slots:
            key = (s["date"], s["startTime"], s["endTime"])
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        uniq.sort(key=lambda s: (s["date"], s["startTime"], s["endTime"]))
        return uniq

    def clear(self):
        """Очищает накопленные слоты"""
        self.accumulated_slots = []

    def reset(self):
        """Алиас для clear()"""
        self.clear()

    def _call_inner(self, image_input: Any) -> List[Dict]:
        """Внутренний метод для вызова обработки"""
        try:
            slots = self.inner.process_image(image_input)
            return slots
        except Exception as e:
            logger.error(f"MemorySlotParser error: {e}", exc_info=True)
            return []

    def process_all_screenshots(self) -> List[Dict]:
        """Для совместимости с SlotParser"""
        return self.get_all_slots()


__all__ = ["SlotParser", "MemorySlotParser", "NewFormatSlotParser"]