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
    Надёжное извлечение дня из жёлтого квадрата, ориентировано на чёрный шрифт на желтом фоне.
    Возвращает int (1..31) или None.
    """
    try:
        x, y, w, h = box
        padding = max(6, int(min(w, h) * 0.18))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)

        roi = img[y1:y2, x1:x2].copy()
        if roi.size == 0:
            logger.warning("Empty ROI for yellow box")
            return None

        # ---------- 1. Предобработка: L-CLAHE и V-inv ----------
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        l_blur = cv2.bilateralFilter(l_channel, d=7, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_blur)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        v_blur = cv2.GaussianBlur(v, (3, 3), 0)
        _, v_inv = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ---------- 2. Выделение вертикальных штрихов (Sobel X -> усиление '1') ----------
        sobelx = cv2.Sobel(l_clahe, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(255.0 * (sobelx / (sobelx.max() + 1e-9)))
        _, sob_mask = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ---------- 3. Комбинация масок и морфология ----------
        # Маска "не-жёлтый" чтобы исключить желтые артефакты
        lower_y = np.array([10, 80, 80])
        upper_y = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_y, upper_y)
        non_yellow = cv2.bitwise_not(yellow_mask)

        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # вертикальное усиление
        kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        combined = cv2.bitwise_or(v_inv, sob_mask)
        combined = cv2.bitwise_and(combined, non_yellow)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_sq, iterations=1)
        combined = cv2.dilate(combined, kernel_v, iterations=1)  # усиливаем тонкие вертикали

        # Еще один вариант: адаптивный порог от L-CLAHE (инвертирован)
        try:
            th_adapt = cv2.adaptiveThreshold(l_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
        except Exception:
            _, th_adapt = cv2.threshold(l_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Подготовим набор изображений для OCR
        variants = {
            "combined": combined,
            "v_inv": v_inv,
            "l_clahe": l_clahe,
            "th_adapt": th_adapt
        }

        # ---------- 4. Подготовка шаблонов цифр для template-matching ----------
        def _make_digit_templates(size=64):
            templates = {}
            for d in range(10):
                tpl = np.ones((size, size), dtype=np.uint8) * 255
                text = str(d)
                # Подобрать масштаб и толщину под UI-цифры
                font_scale = size / 50.0
                thickness = max(1, int(font_scale))
                # центрируем текст
                ((tw, th), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                org = ((size - tw) // 2, (size + th) // 2)
                cv2.putText(tpl, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,), thickness, cv2.LINE_AA)
                _, tpl_bin = cv2.threshold(tpl, 200, 255, cv2.THRESH_BINARY)
                templates[d] = tpl_bin
            return templates

        templates = _make_digit_templates(64)

        # helper: OCR via image_to_data (возвращаем список (num, conf))
        def ocr_candidates_from_image(gray_img, source_name):
            results = []
            h0, w0 = gray_img.shape
            # масштабируем для OCR
            scale = max(1.0, min(4.0, 300.0 / max(h0, w0)))
            img_r = cv2.resize(gray_img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_CUBIC)
            cfgs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789/',
                '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
            ]
            for cfg in cfgs:
                try:
                    data = pytesseract.image_to_data(img_r, config=cfg, output_type=pytesseract.Output.DICT)
                    n = len(data.get("text", []))
                    for i in range(n):
                        txt = (data["text"][i] or "").strip()
                        conf_raw = data.get("conf", [])[i] if i < len(data.get("conf", [])) else '-1'
                        try:
                            conf = int(conf_raw) if str(conf_raw).lstrip('-').isdigit() else -1
                        except:
                            conf = -1
                        if not txt:
                            continue
                        txt_norm = txt.replace(' ', '').replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
                        if '/' in txt_norm:
                            left = txt_norm.split('/')[0]
                            for m in re.findall(r'\d{1,2}', left):
                                val = int(m)
                                if 1 <= val <= 31:
                                    results.append((val, max(conf, 0), f"{source_name}[{cfg}]"))
                        else:
                            for m in re.findall(r'\d{1,2}', txt_norm):
                                val = int(m)
                                if 1 <= val <= 31:
                                    results.append((val, max(conf, 0), f"{source_name}[{cfg}]"))
                except Exception as e:
                    logger.debug(f"OCR fail {source_name} cfg={cfg}: {e}")
            return results

        # helper: template matching for a crop, returns (best_digit, score)
        def template_match_digit(crop_bin):
            # ensure binary 0/255
            _, crop_thr = cv2.threshold(crop_bin, 128, 255, cv2.THRESH_BINARY)
            ch, cw = crop_thr.shape
            best_digit = None
            best_score = -1.0
            for d, tpl in templates.items():
                # resize tpl to crop size or crop to tpl size; we'll resize crop to tpl size
                try:
                    crop_r = cv2.resize(crop_thr, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_AREA)
                    # matchTemplate expects source larger than template; we have equal sizes but it works
                    res = cv2.matchTemplate(255 - crop_r, 255 - tpl, cv2.TM_CCOEFF_NORMED)
                    if res is None:
                        continue
                    score = float(res.max())
                    if score > best_score:
                        best_score = score
                        best_digit = d
                except Exception as e:
                    continue
            return best_digit, best_score

        # ---------- 5. Собираем кандидатов: из OCR по вариантам + connected components ----------
        raw_candidates = []

        for name, var in variants.items():
            if var is None:
                continue
            # приведение к grayscale если нужно
            if len(var.shape) == 3:
                var_gray = cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
            else:
                var_gray = var
            raw_candidates.extend(ocr_candidates_from_image(var_gray, name))

        # connected components на комбинированной маске
        try:
            blobs = combined.copy()
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blobs, connectivity=8)
            comps = []
            for i in range(1, num_labels):
                x_b, y_b, w_b, h_b, area_b = stats[i]
                # допускаем маленькие области (1 может быть тонким)
                if area_b < 8 or w_b < 3 or h_b < 6:
                    continue
                comps.append((x_b, y_b, w_b, h_b, area_b))
            # сортируем слева-направо
            comps.sort(key=lambda r: r[0])
            if len(comps) >= 2:
                # получаем OCR/templatematch для каждой компоненты
                for c in comps:
                    xb, yb, wb, hb, _ = c
                    crop = blobs[yb:yb+hb, xb:xb+wb]
                    # OCR first
                    oc = ocr_candidates_from_image(crop, f"cc_{xb}")
                    if oc:
                        raw_candidates.extend(oc)
                    else:
                        # template match fallback
                        d, score = template_match_digit(crop)
                        if d is not None:
                            # хит: переводим score (0..1) в "conf" шкалу (0..100)
                            raw_candidates.append((d, int(score * 100), f"tmpl_cc_{xb}"))
            elif len(comps) == 1:
                # если один компонент — возможно две цифры внутри. попробуем вертикальную проекцию для сплита
                xb, yb, wb, hb, _ = comps[0]
                crop_full = blobs[yb:yb+hb, xb:xb+wb]
                # вертикальная проекция
                proj = np.sum(crop_full == 255, axis=0)
                if crop_full.shape[1] >= 30:
                    # smooth projection
                    proj_s = cv2.GaussianBlur(proj.reshape(1, -1).astype(np.float32), (1, 5), 0).flatten()
                    # ищем минимум между двух пиков: простой эвристикой
                    peaks = np.where(proj_s > (proj_s.max() * 0.3))[0]
                    if len(peaks) > 0:
                        # попробуем разделить на половины по средней точке между leftmost and rightmost peaks
                        left = np.min(peaks); right = np.max(peaks)
                        if right - left > 10:
                            split = int((left + right) // 2)
                            left_crop = crop_full[:, :split]
                            right_crop = crop_full[:, split:]
                            for idx, sub in enumerate([left_crop, right_crop]):
                                oc = ocr_candidates_from_image(sub, f"split_{idx}")
                                if oc:
                                    raw_candidates.extend(oc)
                                else:
                                    d, score = template_match_digit(sub)
                                    if d is not None:
                                        raw_candidates.append((d, int(score * 100), f"tmpl_split_{idx}"))
                        else:
                            # если не получилось надёжно разделить — пробуем OCR на весь crop
                            oc = ocr_candidates_from_image(crop_full, "cc_full")
                            if oc:
                                raw_candidates.extend(oc)
                            else:
                                d, score = template_match_digit(crop_full)
                                if d is not None:
                                    raw_candidates.append((d, int(score * 100), "tmpl_cc_full"))
                    else:
                        # fallback: OCR на весь bbox
                        oc = ocr_candidates_from_image(crop_full, "cc_full_fb")
                        if oc:
                            raw_candidates.extend(oc)
                        else:
                            d, score = template_match_digit(crop_full)
                            if d is not None:
                                raw_candidates.append((d, int(score * 100), "tmpl_cc_full2"))
                else:
                    oc = ocr_candidates_from_image(crop_full, "cc_small")
                    if oc:
                        raw_candidates.extend(oc)
                    else:
                        d, score = template_match_digit(crop_full)
                        if d is not None:
                            raw_candidates.append((d, int(score * 100), "tmpl_cc_small"))
        except Exception as e:
            logger.debug(f"Connected components pass failed: {e}")

        # ---------- 6. Агрегация кандидатов ----------
        if raw_candidates:
            score_map = {}
            for num, conf, src in raw_candidates:
                # conf уже в 0..100 (OCR) или template mapped to 0..100
                score_map.setdefault(num, 0)
                score_map[num] += max(0, int(conf))
            # приоритет двузначных (10..31)
            sorted_nums = sorted(score_map.items(), key=lambda kv: (- (10 <= kv[0] <= 31), -kv[1], -kv[0]))
            chosen = sorted_nums[0][0]
            logger.info(f"Day candidates (agg): {score_map}, selected: {chosen}")
            return chosen

        # ---------- 7. Последняя попытка: OCR без whitelist на L-CLAHE и v_inv ----------
        try:
            txt = pytesseract.image_to_string(l_clahe, config='--oem 3 --psm 6')
            if txt:
                m = re.findall(r'\b([12]?\d|3[01])\b', txt)
                if m:
                    v = int(m[0])
                    if 1 <= v <= 31:
                        logger.info(f"Fallback found day from l_clahe: {v}")
                        return v
            txt2 = pytesseract.image_to_string(v_inv, config='--oem 3 --psm 6')
            if txt2:
                m = re.findall(r'\b([12]?\d|3[01])\b', txt2)
                if m:
                    v = int(m[0])
                    if 1 <= v <= 31:
                        logger.info(f"Fallback found day from v_inv: {v}")
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