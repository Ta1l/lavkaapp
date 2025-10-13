# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Версия, переписанная для использования EasyOCR вместо Tesseract.

Сохраняет прежний API и поведение класса/функций, но заменяет движок OCR на EasyOCR
и использует гибридный подход для надежного извлечения дня из жёлтого квадрата.
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

# Попытка импортировать easyocr
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None
    _EASYOCR_AVAILABLE = False

logger = logging.getLogger("lavka.ocr_module")
logger.setLevel(logging.INFO)

# --------- Константы ---------
FIXED_YEAR = 2025
FIXED_MONTH = 10

# Паттерны для поиска времени
TIME_RANGE_PATTERNS = [
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    re.compile(r"(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})"),
]

# Инициализация EasyOCR reader (один раз)
def _init_reader(langs: List[str] = None) -> Optional["easyocr.Reader"]:
    if not _EASYOCR_AVAILABLE:
        logger.error("easyocr is not installed; install it with `pip install easyocr`.")
        return None
    if langs is None:
        langs = ["ru", "en"]
    # Попытаемся определить, доступна ли GPU
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False
    try:
        reader = easyocr.Reader(langs, gpu=use_gpu)
        logger.info(f"EasyOCR reader initialized (gpu={use_gpu}) for langs={langs}")
        return reader
    except Exception as e:
        logger.error(f"Failed to init EasyOCR reader: {e}")
        return None

# Создаем reader глобально — позже можно переопределить при тестах
_READER = _init_reader()

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
            if 0.7 <= aspect_ratio <= 1.5:
                valid_boxes.append((x, y, w, h, area))
        if not valid_boxes:
            return None
        valid_boxes.sort(key=lambda b: b[4], reverse=True)
        best = valid_boxes[0][:4]
        logger.info(f"Found yellow box at ({best[0]}, {best[1]}) size ({best[2]}x{best[3]})")
        return best
    except Exception as e:
        logger.error(f"Error finding yellow box: {e}")
        return None


def _extract_day_simple(img: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[int]:
    """
    Улучшенное извлечение дня (EasyOCR + template) с улучшенной агрегацией и штрафами для шаблонных совпадений.
    Возвращает int 1..31 или None.
    """
    global _READER
    try:
        if _READER is None:
            logger.error("EasyOCR reader is not initialized. Cannot extract day.")
            return None

        x, y, w, h = box
        pad = max(0, int(min(w, h) * 0.05))
        x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        roi_bgr = img[y1:y2, x1:x2].copy()
        if roi_bgr.size == 0:
            logger.warning("Empty ROI for yellow box")
            return None

        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # 1) EasyOCR candidates (several scales)
        ocr_tokens = []  # list of dicts: {"num":int, "conf":float(0..100), "x":int, "src":"ocr"}
        scales = [1.5, 2.5, 3.5]
        for s in scales:
            try:
                h0, w0 = roi_gray.shape
                img_r = cv2.resize(roi_gray, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_CUBIC)
                results = _READER.readtext(img_r, detail=1, paragraph=False)
            except Exception as e:
                logger.debug(f"EasyOCR readtext failed on scale={s}: {e}")
                results = []
            for bbox, text, conf in results:
                if not text:
                    continue
                tnorm = text.replace(' ', '').replace('O','0').replace('o','0').replace('l','1').replace('I','1')
                if '/' in tnorm:
                    matches = re.findall(r'\d{1,2}', tnorm.split('/')[0])
                else:
                    matches = re.findall(r'\d{1,2}', tnorm)
                try:
                    xs = [int(pt[0]) for pt in bbox]
                    left_px = min(xs)
                    x_coord = int(left_px / s)
                except Exception:
                    x_coord = None
                for m in matches:
                    v = int(m)
                    if 1 <= v <= 31:
                        ocr_tokens.append({"num": v, "conf": float(conf) * 100.0, "x": x_coord, "src": "ocr"})

        # 2) Template matching as before (edge templates)
        edges = cv2.Canny(roi_gray, 50, 150)
        def _make_edge_templates(sizes=(28,36,48)):
            tpls = {}
            for size in sizes:
                tpl_dict = {}
                for d in range(10):
                    canvas = np.ones((size, size), dtype=np.uint8) * 255
                    font_scale = size / 40.0
                    thickness = max(1, int(font_scale))
                    ((tw, th), _) = cv2.getTextSize(str(d), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    org = ((size - tw) // 2, (size + th) // 2)
                    cv2.putText(canvas, str(d), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,), thickness, cv2.LINE_AA)
                    tpl_edge = cv2.Canny(canvas, 50, 150)
                    tpl_dict[d] = tpl_edge
                tpls[size] = tpl_dict
            return tpls
        templates = _make_edge_templates((28,36,48))

        def _nms_matches(score_map, tpl_w, tpl_h, thresh=0.55):
            matches = []
            sm = score_map.copy()
            while True:
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(sm)
                if maxVal < thresh:
                    break
                matches.append((maxLoc[0], maxLoc[1], float(maxVal)))
                x0, y0 = maxLoc
                x1n = max(0, x0 - tpl_w//2); y1n = max(0, y0 - tpl_h//2)
                x2n = min(sm.shape[1], x0 + tpl_w//2); y2n = min(sm.shape[0], y0 + tpl_h//2)
                sm[y1n:y2n, x1n:x2n] = 0
            return matches

        template_tokens = []  # list of dicts: {"num":int, "conf":int(0..100), "x":int, "src":"tmpl"}
        for size, tpl_dict in templates.items():
            for d, tpl_edge in tpl_dict.items():
                try:
                    if edges.shape[0] < tpl_edge.shape[0] or edges.shape[1] < tpl_edge.shape[1]:
                        continue
                    res = cv2.matchTemplate(edges, tpl_edge, cv2.TM_CCOEFF_NORMED)
                    matches = _nms_matches(res, tpl_edge.shape[1], tpl_edge.shape[0], thresh=0.58)
                    for mx, my, score in matches:
                        template_tokens.append({"num": int(d), "conf": int(score * 100.0), "x": int(mx), "src": "tmpl"})
                except Exception:
                    continue
        # fallback lower threshold
        if not template_tokens:
            for size, tpl_dict in templates.items():
                for d, tpl_edge in tpl_dict.items():
                    try:
                        if edges.shape[0] < tpl_edge.shape[0] or edges.shape[1] < tpl_edge.shape[1]:
                            continue
                        res = cv2.matchTemplate(edges, tpl_edge, cv2.TM_CCOEFF_NORMED)
                        matches = _nms_matches(res, tpl_edge.shape[1], tpl_edge.shape[0], thresh=0.45)
                        for mx, my, score in matches:
                            template_tokens.append({"num": int(d), "conf": int(score * 100.0), "x": int(mx), "src": "tmpl"})
                    except Exception:
                        continue

        merged_tokens = []
        for t in ocr_tokens:
            merged_tokens.append(t)
        for t in template_tokens:
            merged_tokens.append(t)

        if not merged_tokens:
            logger.warning("No tokens found by EasyOCR or templates in ROI.")
            return None

        # Build contributions: number -> list of contributions (conf, src, x)
        contributions = {}  # key: candidate_number (1..31 or composed) -> list of dicts
        # first: single-digit contributions
        for tok in merged_tokens:
            n = int(tok["num"])
            contributions.setdefault(n, []).append({"conf": int(tok["conf"]), "src": tok.get("src", "unk"), "x": tok.get("x")})

        # try to form two-digit candidates from tokens_with_x by proximity
        tokens_with_x = [t for t in merged_tokens if t.get("x") is not None]
        if tokens_with_x:
            tokens_with_x.sort(key=lambda t: t["x"])
            roi_w = roi_gray.shape[1]
            for i in range(len(tokens_with_x)):
                left = tokens_with_x[i]
                for j in (i+1, i+2):
                    if j >= len(tokens_with_x):
                        break
                    right = tokens_with_x[j]
                    dist = right["x"] - left["x"]
                    # tighter distance threshold than раньше: не больше 0.5 * roi_w and absolute < 60 px
                    if dist < max(min(roi_w * 0.5, 60), 30):
                        num = left["num"] * 10 + right["num"]
                        if 1 <= num <= 31:
                            contributions.setdefault(num, [])
                            contributions[num].append({"conf": int(left["conf"]), "src": left.get("src","unk"), "x": left.get("x")})
                            contributions[num].append({"conf": int(right["conf"]), "src": right.get("src","unk"), "x": right.get("x")})

        # Now compute metrics for each candidate
        candidate_metrics = {}  # num -> metrics dict
        # source weights
        W = {"ocr": 1.0, "tmpl": 0.6, "unk": 0.5}

        for num, contribs in contributions.items():
            # deduplicate contributions (same x and src) optionally
            # compute totals
            total_weighted = 0.0
            total_conf = 0.0
            xs = []
            count_ocr = 0
            for c in contribs:
                src = c.get("src","unk")
                weight = W.get(src, 0.6)
                conf = int(c.get("conf", 0))
                total_weighted += conf * weight
                total_conf += conf
                if c.get("x") is not None:
                    xs.append(int(c["x"]))
                if src == "ocr":
                    count_ocr += 1
            count = len(contribs)
            avg_conf = (total_conf / count) if count > 0 else 0.0
            span = (max(xs) - min(xs)) if xs else 0
            # penalize huge spatial span (digits too far apart)
            span_penalty = 1.0
            if span > max(roi_gray.shape[1] * 0.6, 80):
                span_penalty = 0.6
            # validity checks for two-digit: require at least one OCR token or both template tokens very confident
            is_two_digit = (10 <= num <= 31)
            valid_two_digit = True
            if is_two_digit:
                if count_ocr >= 1:
                    valid_two_digit = True
                else:
                    # if no OCR contributors, require both parts to have high template conf (>=85)
                    # identify approx left and right confs if available by splitting contribs by x
                    if len(xs) >= 2:
                        # basic split
                        sorted_cs = sorted([c for c in contribs if c.get("x") is not None], key=lambda c: c["x"])
                        # estimate two groups
                        left_conf = sorted_cs[0]["conf"] if len(sorted_cs) >= 1 else 0
                        right_conf = sorted_cs[1]["conf"] if len(sorted_cs) >= 2 else 0
                        if not (left_conf >= 85 and right_conf >= 85):
                            valid_two_digit = False
                    else:
                        # if not enough spatial info, be conservative
                        valid_two_digit = False
            # final_score after span penalty
            final_score = total_weighted * span_penalty
            candidate_metrics[num] = {
                "total_weighted": total_weighted,
                "avg_conf": avg_conf,
                "count": count,
                "count_ocr": count_ocr,
                "span": span,
                "span_penalty": span_penalty,
                "valid_two_digit": valid_two_digit,
                "final_score": final_score
            }

        # Logging detailed metrics in debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Candidate metrics detail:")
            for num, m in candidate_metrics.items():
                logger.debug(f"  num={num}: {m}")

        # Filter out invalid two-digit candidates (if any)
        filtered_candidates = {}
        for num, m in candidate_metrics.items():
            if 10 <= num <= 31 and not m["valid_two_digit"]:
                # penalize heavily: reduce final_score
                m["final_score"] *= 0.4
            filtered_candidates[num] = m["final_score"]

        if not filtered_candidates:
            logger.warning("No candidates after filtering")
            return None

        # Choose best: prefer valid two-digit numbers, then by final_score, then avg_conf
        # create sortable list with (is_two_digit_valid_flag, final_score, avg_conf, num)
        sortable = []
        for num, score in filtered_candidates.items():
            m = candidate_metrics[num]
            is_two_valid = 1 if (10 <= num <= 31 and m["valid_two_digit"]) else 0
            sortable.append((is_two_valid, m["final_score"], m["avg_conf"], num))
        # sort by: is_two_valid desc, final_score desc, avg_conf desc, num desc
        sortable.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3]))

        chosen_num = int(sortable[0][3])
        # log human friendly breakdown
        logger.info(f\"Day candidates (detailed): { {k: candidate_metrics[k] for k in sorted(candidate_metrics.keys())} }\")\n        logger.info(f\"Selected day: {chosen_num}\")\n        return chosen_num

    except Exception as e:
        logger.error(f"Error extracting day with enhanced aggregation: {e}")
        return None



def _find_time_slots(img: np.ndarray) -> List[Tuple[str, str]]:
    """
    Находит временные слоты на изображении используя EasyOCR.
    """
    global _READER
    try:
        if _READER is None:
            logger.error("EasyOCR reader is not initialized. Cannot find time slots.")
            return []

        # Берем нижние 3/4 изображения
        h = img.shape[0]
        bottom_part = img[h//4:, :]
        gray = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2GRAY)

        # Используем reader.readtext
        try:
            results = _READER.readtext(gray, detail=1, paragraph=False)
        except Exception as e:
            logger.debug(f"EasyOCR readtext failed on time area: {e}")
            results = []

        # Собираем весь текст в строку с позициями
        text_lines = []
        for bbox, text, conf in results:
            if not text:
                continue
            t = text.replace('О', '0').replace('о', '0').replace('З', '3').replace('з', '3')
            text_lines.append(t)

        full_text = "\n".join(text_lines)
        slots = []
        for pattern in TIME_RANGE_PATTERNS:
            matches = pattern.findall(full_text)
            for match in matches:
                try:
                    h1, m1, h2, m2 = match
                    h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
                    if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                        if h2 > h1 or (h2 == h1 and m2 > m1):
                            start_time = f"{h1:02d}:{m1:02d}"
                            end_time = f"{h2:02d}:{m2:02d}"
                            slots.append((start_time, end_time))
                except Exception:
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
        logger.error(f"Error finding time slots with EasyOCR: {e}")
        return []


# --------- Основной класс парсера ---------
class NewFormatSlotParser:
    """Парсер для скриншотов с желтым выделением дня."""
    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

    def process_image(self, image_input: Any) -> List[Dict]:
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

        # 3. Извлекаем день ( EasyOCR метод )
        day = _extract_day_simple(img, yellow_box)

        if day is None:
            logger.error("Could not extract day from yellow box")
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


# --------- Классы MemorySlotParser и SlotParser (без изменений по API) ---------
class MemorySlotParser:
    """Парсер для обработки скриншотов из памяти."""
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.parser = NewFormatSlotParser(debug=debug)
        self.accumulated_slots = []
        self.screenshots_count = 0
        self.base_path = ""
        self.cancelled_count = 0
        self.last_error = None

    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
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
        return self.process_screenshot_from_memory(image_bytes, is_last)

    def process_image(self, image_input: Any) -> List[Dict]:
        try:
            return self.parser.process_image(image_input)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error: {e}")
            return []

    def process_image_bytes(self, image_bytes: bytes) -> List[Dict]:
        try:
            return self.parser.process_image(image_bytes)
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error: {e}")
            return []

    def get_all_slots(self) -> List[Dict]:
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
        self.accumulated_slots = []
        self.screenshots_count = 0
        self.last_error = None
        self.cancelled_count = 0

    def reset(self):
        self.clear()

    def preprocess_image_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        return gray

    def _extract_lines_from_data(self, data: Dict) -> List[Dict]:
        return []

    def process_all_screenshots(self) -> List[Dict]:
        return self.get_all_slots()


class SlotParser:
    """Парсер для папки со скриншотами."""
    def __init__(self, base_path: str, debug: bool = False):
        self.base_path = base_path
        self.parser = NewFormatSlotParser(debug=debug)
        self.cancelled_count = 0

    def process_all_screenshots(self) -> List[Dict]:
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
