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
    Кардинально новый подход:
      - вырезаем ROI (ЖЁЛТЫЙ квадратик) без цветовых преобразований
      - масштабируем ROI и запускаем pytesseract.image_to_data на нескольких масштабах
      - делаем template-matching по Canny-границам исходного ROI (несколько размеров шаблонов)
      - собираем одиночные и парные (двузначные) кандидаты, агрегируем по score и выбираем лучший
    Возвращает int 1..31 или None.
    """
    try:
        x, y, w, h = box
        # crop EXACT (минимальный padding = 0) — ты просил не менять изображение
        x1, y1, x2, y2 = x, y, x + w, y + h
        roi_bgr = img[y1:y2, x1:x2].copy()
        if roi_bgr.size == 0:
            logger.warning("Empty ROI for yellow box")
            return None

        # convert to gray (but DO NOT apply heavy binarization)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # --- 1) OCR на нескольких масштабах (используем image_to_data чтобы получить позиции/conf) ---
        ocr_tokens = []  # each -> {"num": int, "conf": int(0..100), "x": int}
        scales = [1.8, 2.5, 3.5]  # увеличиваем, т.к. OCR любит большие цифры

        for s in scales:
            h0, w0 = roi_gray.shape
            img_r = cv2.resize(roi_gray, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_CUBIC)
            cfgs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789/',
                '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/'  # line
            ]
            for cfg in cfgs:
                try:
                    data = pytesseract.image_to_data(img_r, config=cfg, output_type=pytesseract.Output.DICT)
                except Exception as e:
                    logger.debug(f"pytesseract failed for scale={s} cfg={cfg}: {e}")
                    continue
                n = len(data.get("text", []))
                for i in range(n):
                    txt = (data["text"][i] or "").strip()
                    if not txt:
                        continue
                    # normalize common conf forms
                    conf_raw = data.get("conf", [])[i] if i < len(data.get("conf", [])) else '-1'
                    try:
                        conf = int(conf_raw) if str(conf_raw).lstrip('-').isdigit() else -1
                    except:
                        conf = -1
                    # left coordinate in scaled image -> map back to ROI coords
                    left = data.get("left", [None])[i]
                    if left is not None:
                        x_coord = int(left / s)
                    else:
                        x_coord = None
                    # normalize text and extract digits (handle '/')
                    tnorm = txt.replace(' ', '').replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
                    if '/' in tnorm:
                        left_part = tnorm.split('/')[0]
                        m = re.findall(r'\d{1,2}', left_part)
                    else:
                        m = re.findall(r'\d{1,2}', tnorm)
                    for mg in m:
                        v = int(mg)
                        if 1 <= v <= 31:
                            ocr_tokens.append({"num": v, "conf": max(0, conf), "x": x_coord})

        # --- 2) Template-matching on edges (Canny) of original ROI (no color change) ---
        # Prepare edge map of ROI (works well with anti-aliased UI fonts)
        edges = cv2.Canny(roi_gray, 50, 150)
        # Make digit templates as edge templates (draw digits then edge)
        def _make_edge_templates(sizes=(28, 36, 48)):
            tpls = {}
            for size in sizes:
                tpl_dict = {}
                # create square canvas
                for d in range(10):
                    canvas = np.ones((size, size), dtype=np.uint8) * 255
                    # choose font scale relative to size
                    font_scale = size / 40.0
                    thickness = max(1, int(font_scale))
                    ((tw, th), _) = cv2.getTextSize(str(d), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    org = ((size - tw) // 2, (size + th) // 2)
                    cv2.putText(canvas, str(d), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,), thickness, cv2.LINE_AA)
                    # edge of template
                    tpl_edge = cv2.Canny(canvas, 50, 150)
                    tpl_dict[d] = tpl_edge
                tpls[size] = tpl_dict
            return tpls

        templates = _make_edge_templates((28, 36, 48))

        # Non-max suppression helper for matches
        def _nms_matches(score_map, tpl_w, tpl_h, thresh=0.55):
            matches = []
            sm = score_map.copy()
            # iterate finding peaks
            while True:
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(sm)
                if maxVal < thresh:
                    break
                matches.append((maxLoc[0], maxLoc[1], float(maxVal)))
                # zero out neighborhood
                x0, y0 = maxLoc
                x1 = max(0, x0 - tpl_w // 2)
                y1 = max(0, y0 - tpl_h // 2)
                x2 = min(sm.shape[1], x0 + tpl_w // 2)
                y2 = min(sm.shape[0], y0 + tpl_h // 2)
                sm[y1:y2, x1:x2] = 0
            return matches

        template_matches = []  # items: {"digit":d, "x":x, "score":score}
        # match multi-size templates
        for size, tpl_dict in templates.items():
            for d, tpl_edge in tpl_dict.items():
                try:
                    # we need ROI edges larger than template; if smaller, skip
                    if edges.shape[0] < tpl_edge.shape[0] or edges.shape[1] < tpl_edge.shape[1]:
                        continue
                    res = cv2.matchTemplate(edges, tpl_edge, cv2.TM_CCOEFF_NORMED)
                    # find peaks via simple NMS
                    matches = _nms_matches(res, tpl_edge.shape[1], tpl_edge.shape[0], thresh=0.58)
                    for (mx, my, score) in matches:
                        # mx is left in ROI coords for that match
                        template_matches.append({"digit": d, "x": int(mx), "score": float(score)})
                except Exception:
                    continue

        # If no template matches with high threshold, lower threshold and allow more
        if not template_matches:
            for size, tpl_dict in templates.items():
                for d, tpl_edge in tpl_dict.items():
                    try:
                        if edges.shape[0] < tpl_edge.shape[0] or edges.shape[1] < tpl_edge.shape[1]:
                            continue
                        res = cv2.matchTemplate(edges, tpl_edge, cv2.TM_CCOEFF_NORMED)
                        matches = _nms_matches(res, tpl_edge.shape[1], tpl_edge.shape[0], thresh=0.45)
                        for (mx, my, score) in matches:
                            template_matches.append({"digit": d, "x": int(mx), "score": float(score)})
                    except Exception:
                        continue

        # convert template match score to 0..100 scale like OCR conf
        for tm in template_matches:
            tm["conf"] = int(tm["score"] * 100)
            tm["num"] = int(tm["digit"])
            # keep same shape as OCR tokens
            # some templates might overlap the same digit multiple times

        # Merge OCR tokens and template tokens into a single token list
        merged_tokens = []
        for t in ocr_tokens:
            merged_tokens.append({"num": int(t["num"]), "conf": int(t["conf"]), "x": t.get("x")})
        for tm in template_matches:
            # template matches may be many; we add them too
            merged_tokens.append({"num": int(tm["num"]), "conf": int(tm["conf"]), "x": int(tm["x"])})

        # If nothing found at all -> return None
        if not merged_tokens:
            logger.warning("No tokens found by OCR or template matching in ROI (original-crop strategy).")
            return None

        # --- Build multi-digit candidates from tokens using X coordinate (left->right) ---
        # Separate tokens with known x and without
        tokens_with_x = [t for t in merged_tokens if t.get("x") is not None]
        tokens_without_x = [t for t in merged_tokens if t.get("x") is None]

        # Base scoring map (single digit contributions)
        score_map = {}
        for t in merged_tokens:
            score_map.setdefault(t["num"], 0)
            score_map[t["num"]] += max(0, int(t["conf"]))

        # Combine left->right neighbors into two-digit numbers
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
                    # require that two digits are reasonably close (not across entire ROI)
                    if dist < max(roi_w * 0.7, 80):
                        num = left["num"] * 10 + right["num"]
                        if 1 <= num <= 31:
                            # scoring: sum conf
                            ssum = int(left["conf"]) + int(right["conf"])
                            # if either was template-based (high conf) we keep it
                            score_map.setdefault(num, 0)
                            score_map[num] += max(0, ssum)

        # Also try to construct number by ordering template matches even if OCR provided single digit
        # (covering case where OCR gives '1' and template finds '7' a bit to the right)
        if template_matches:
            tms = sorted(template_matches, key=lambda m: m["x"])
            for i in range(len(tms)):
                for j in range(i+1, min(i+3, len(tms))):
                    left = tms[i]; right = tms[j]
                    dist = right["x"] - left["x"]
                    if dist < max(roi_w * 0.7, 80):
                        num = left["num"] * 10 + right["num"]
                        if 1 <= num <= 31:
                            ssum = int(left["conf"]) + int(right["conf"])
                            score_map.setdefault(num, 0)
                            score_map[num] += max(0, ssum)

        # Final selection: prefer two-digit numbers in 10..31, then by highest total score
        if score_map:
            # sort: prefer two-digit 10..31, then by score desc, then numeric desc
            items = list(score_map.items())
            items.sort(key=lambda kv: (- (10 <= kv[0] <= 31), -kv[1], -kv[0]))
            chosen_num = items[0][0]
            logger.info(f"Day candidates (agg): {score_map}, selected: {chosen_num}")
            return chosen_num

        logger.warning("No valid day composed after aggregation on original crop.")
        return None

    except Exception as e:
        logger.error(f"Error in new extract_day_simple(original-crop): {e}")
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