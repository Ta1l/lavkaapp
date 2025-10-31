# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Ensemble-пайплайн (Tesseract + template) + более строгая логика fallback'а.
Сохраняет прежний API и структуру.
"""

import os
import re
import logging
import math
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

# logger
logger = logging.getLogger("lavka.ocr_module")
logger.setLevel(logging.INFO)

# pytesseract init (требует, чтобы путь был задан до импорта, если нужно)
try:
    import pytesseract
    if not getattr(pytesseract.pytesseract, "tesseract_cmd", None):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    _PYTESSERACT_AVAILABLE = True
    logger.info(f"Tesseract initialized at: {pytesseract.pytesseract.tesseract_cmd}")
except Exception as e:
    pytesseract = None
    _PYTESSERACT_AVAILABLE = False
    logger.error(f"Tesseract initialization failed: {e}")

# constants
FIXED_YEAR = 2025
FIXED_MONTH = 10

TIME_RANGE_PATTERNS = [
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    re.compile(r"(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})"),
]

# ---------- helpers ----------
def _read_image(image_input: Any) -> Optional[np.ndarray]:
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                logger.error(f"File not found: {image_input}")
                return None
            return cv2.imread(image_input)
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if isinstance(image_input, BytesIO):
            image_input.seek(0)
            data = image_input.read()
            nparr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        logger.error(f"Unknown image type: {type(image_input)}")
        return None
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return None

# templates (edge)
def _make_templates(sizes=(28,36,48)):
    tpls = {}
    for size in sizes:
        digit_map = {}
        for d in range(10):
            canvas = np.ones((size, size), dtype=np.uint8) * 255
            font_scale = size / 30.0
            thickness = max(2, int(font_scale))
            ((tw, th), _) = cv2.getTextSize(str(d), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            org = ((size - tw)//2, (size + th)//2)
            cv2.putText(canvas, str(d), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,), thickness, cv2.LINE_AA)
            edge = cv2.Canny(canvas, 50, 150)
            digit_map[d] = edge
        tpls[size] = digit_map
    return tpls

_TEMPLATES = _make_templates((28,36,48))

def _edge_similarity(a: np.ndarray, b: np.ndarray) -> float:
    try:
        a_f = a.astype(np.float32)/255.0
        b_f = b.astype(np.float32)/255.0
        dist = np.linalg.norm(a_f - b_f)
        max_norm = math.sqrt(a.shape[0]*a.shape[1]) + 1e-6
        sim = max(0.0, 1.0 - dist/max_norm)
        return sim
    except Exception:
        return 0.0

def _classify_digit_patch(patch_gray: np.ndarray) -> Tuple[int, float]:
    try:
        edges = cv2.Canny(patch_gray, 50, 150)
        h, w = edges.shape
        best_d, best_score = -1, 0.0
        for size, digit_map in _TEMPLATES.items():
            maxi = max(h,w,1)
            canvas = np.zeros((maxi, maxi), dtype=np.uint8)
            y_off = (maxi - h)//2
            x_off = (maxi - w)//2
            canvas[y_off:y_off+h, x_off:x_off+w] = edges
            resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)
            for d, tpl in digit_map.items():
                s = _edge_similarity(resized, tpl)
                score = s * 100.0
                if score > best_score:
                    best_score = score
                    best_d = d
        if best_score < 30.0:
            return (-1, 0.0)
        return (best_d, best_score)
    except Exception as e:
        logger.debug(f"classify_digit_patch error: {e}")
        return (-1, 0.0)

def _classify_digit_patch_from_gray(patch_gray: np.ndarray) -> Tuple[int, float]:
    try:
        h,w = patch_gray.shape
        if h < 6 or w < 6:
            factor = max(2, int(6.0 / min(h,w)))
            patch_gray = cv2.resize(patch_gray, (w*factor, h*factor), interpolation=cv2.INTER_CUBIC)
        patch_gray = cv2.GaussianBlur(patch_gray, (3,3), 0)
        patch_gray = cv2.normalize(patch_gray, None, 0, 255, cv2.NORM_MINMAX)
        return _classify_digit_patch(patch_gray)
    except Exception as e:
        logger.debug(f"_classify_digit_patch_from_gray error: {e}")
        return (-1, 0.0)

# find yellow box (tuned)
def _find_yellow_box(img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([6, 50, 60])
        upper = np.array([45, 255, 255])
        mask1 = cv2.inRange(hsv, lower, upper)
        h_channel = hsv[:,:,0]
        v_channel = hsv[:,:,2]
        mask2 = cv2.inRange(h_channel, 6, 45)
        mask2 = cv2.bitwise_and(mask2, cv2.inRange(v_channel, 80, 255))
        mask = cv2.bitwise_or(mask1, mask2)
        H = mask.shape[0]
        mask[H//2:, :] = 0
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x,y,w,hcnt = cv2.boundingRect(cnt)
            ar = w / float(hcnt) if hcnt>0 else 0
            if 0.4 <= ar <= 2.5:
                valid.append((x,y,w,hcnt,area))
        if not valid:
            return None
        valid.sort(key=lambda x: x[4], reverse=True)
        x,y,w,hcnt,_ = valid[0]
        pad = max(3, int(0.08 * max(w,hcnt)))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad); y2 = min(img.shape[0], y + hcnt + pad)
        logger.info(f"Found yellow box at ({x1}, {y1}) size ({x2-x1}x{y2-y1})")
        return (x1, y1, x2-x1, y2-y1)
    except Exception as e:
        logger.error(f"Error finding yellow box: {e}")
        return None

# ensemble extractor (improvements + safer fallback)
# ensemble extractor (improvements + safer fallback)
def _extract_day_ensemble(img: np.ndarray, box: Tuple[int,int,int,int], debug: bool=False) -> Optional[int]:
    try:
        x, y, w, h = box
        if w <=0 or h<=0:
            return None
        pad = 2
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad); y2 = min(img.shape[0], y + h + pad)
        roi = img[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # upscale
        scale = 4
        gray_large = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # reduce yellow influence: replace yellow pixels with WHITE
        try:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_y = np.array([6,50,60]); upper_y = np.array([45,255,255])
            mask_y = cv2.inRange(hsv_roi, lower_y, upper_y)
            mask_y_large = cv2.resize(mask_y, (gray_large.shape[1], gray_large.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # ----------------------------------------------------------------
            # !!! ГЛАВНЫЙ ФИКС !!!
            # Заменяем желтый фон (mask_y_large > 0) на БЕЛЫЙ (255)
            # Это создает высокий контраст (темная цифра на белом фоне)
            gray_large[mask_y_large > 0] = 255
            # ----------------------------------------------------------------
            
            # --- СТАРАЯ ЛОГИКА (ОШИБОЧНАЯ) ---
            # non_y_vals = gray_large[mask_y_large==0]
            # if non_y_vals.size > 0:
            #     rep = int(np.median(non_y_vals))
            # else:
            #     rep = int(np.median(gray_large))
            # gray_large[mask_y_large>0] = rep
            # --- КОНЕЦ СТАРОЙ ЛОГИКИ ---
            
        except Exception as e:
            logger.debug(f"Yellow reduction failed, proceeding anyway: {e}")
            pass

        gray_large = cv2.equalizeHist(gray_large)
        gray_large = cv2.GaussianBlur(gray_large, (3,3), 0)

        # create binary variants
        bin_variants = []
        _, b_otsu_inv = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bin_variants.append(('otsu_inv', b_otsu_inv))
        
        _, b_otsu = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_variants.append(('otsu', b_otsu)) # Этот вариант теперь должен быть лучшим
        
        for t in (90,110,130,150):
            _, bt = cv2.threshold(gray_large, t, 255, cv2.THRESH_BINARY_INV)
            bin_variants.append((f'inv_{t}', bt))
        try:
            # НОВОЕ: Добавляем обычный (не-инвертированный) адаптивный порог,
            # который хорош для темного текста на светлом фоне.
            adapt = cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            bin_variants.append(('adapt', adapt))
            
            adapt_inv = cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            bin_variants.append(('adapt_inv', adapt_inv))
        except Exception:
            pass

        # try pytesseract on bin_variants
        tesseract_candidates = []
        if _PYTESSERACT_AVAILABLE:
            # УЛУЧШЕНИЕ: Добавляем psm 10 (один символ) и psm 6 (один блок)
            configs = [
                '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789', # 1. Искать один символ
                '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',  # 2. Искать одно слово
                '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',  # 3. Искать одну строку
                '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',  # 4. Искать один блок
            ]
            for vname, bin_img in bin_variants:
                try:
                    pil = Image.fromarray(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB))
                except Exception:
                    try:
                        pil = Image.fromarray(bin_img)
                    except Exception:
                        continue
                for cfg in configs:
                    try:
                        data = pytesseract.image_to_data(pil, config=cfg, output_type=pytesseract.Output.DICT)
                        texts = data.get('text', [])
                        confs = data.get('conf', [])
                        combined = ''.join([t for t in texts if t and t.strip()])
                        filtered = re.sub(r'\D','', combined)
                        max_conf = -1.0
                        for t,c in zip(texts, confs):
                            if t and re.search(r'\d', t):
                                try:
                                    cf = float(c)
                                except Exception:
                                    cf = -1.0
                                if cf > max_conf:
                                    max_conf = cf
                        if filtered:
                            try:
                                val = int(filtered)
                                if 1 <= val <= 31:
                                    tesseract_candidates.append((val, max_conf, cfg, vname))
                            except:
                                pass
                    except Exception:
                        continue

        if tesseract_candidates:
            tesseract_candidates.sort(key=lambda x:(x[1], -abs(x[0]-datetime.now().day)), reverse=True)
            best_val,best_conf,cfg_name,v_name = tesseract_candidates[0]
            logger.debug(f"Tesseract candidate: {best_val} conf={best_conf} (cfg={cfg_name}, v={v_name})")
            # Поднимаем порог уверенности, т.к. качество картинки стало выше
            if best_conf >= 70: 
                logger.info(f"Tesseract success: {best_val} (conf={best_conf})")
                return int(best_val)

        # segmentation fallback: 
        # Эта логика должна остаться на THRESH_BINARY_INV,
        # т.к. findContours ищет БЕЛЫЕ объекты (цифры) на ЧЕРНОМ фоне (0).
        # Наш gray_large - темный на белом.
        # THRESH_BINARY_INV сделает темный (ниже порога) -> 255, а белый (выше порога) -> 0.
        # Это корректно.
        _, binary = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = binary.shape
        rects = []
        for cnt in contours:
            x_c,y_c,w_c,h_c = cv2.boundingRect(cnt)
            area = w_c*h_c
            if area < 30 or h_c < 6:
                continue
            if h_c > 0.9 * H: # Слишком высокий
                continue
            if w_c > 0.9 * W: # Слишком широкий
                continue
            rects.append((x_c,y_c,w_c,h_c))
            
        if rects:
            rects = _merge_rects(rects)
            # keep up to 3, sort by x
            rects.sort(key=lambda r: r[0])
            # choose rects closest to center X
            cx = W // 2
            rects = sorted(rects, key=lambda r: abs((r[0]+r[2]/2) - cx))
            rects = rects[:3]
            rects = sorted(rects, key=lambda r: r[0])
            votes = []
            confs = []
            for rx,ry,rw,rh in rects:
                # Берем патч из бинарного (уже очищенного) изображения
                patch_bin = binary[ry:ry+rh, rx:rx+rw]
                d,s = _classify_digit_patch_from_gray(patch_bin) # _classify_digit_patch_from_gray ожидает gray
                votes.append(d); confs.append(s)
            
            # aggregate
            digits = "".join(str(d) if d>=0 else "?" for d in votes)
            digits_clean = re.sub(r'[^0-9]','', digits)
            if digits_clean:
                try:
                    val = int(digits_clean)
                    avg_conf = sum([c for c in confs if c>0])/max(1, len([c for c in confs if c>0]))
                    # if result too large (e.g. >31) try last two digits
                    if not (1 <= val <= 31):
                        if len(digits_clean) >= 2:
                            val_try = int(digits_clean[-2:])
                            if 1 <= val_try <= 31:
                                val = val_try
                                logger.debug(f"Used last-two heuristic -> {val}")
                    if 1 <= val <= 31 and avg_conf >= 30.0:
                        logger.info(f"Template fallback success: {val} (avg_conf={avg_conf})")
                        return val
                except Exception:
                    pass

        # УЛУЧШЕНИЕ: final quick try на БОЛЬШОМ, ОБРАБОТАННОМ gray_large
        if _PYTESSERACT_AVAILABLE:
            try:
                # Используем gray_large, а не старый 'gray'
                large_pil = Image.fromarray(cv2.cvtColor(gray_large, cv2.COLOR_GRAY2RGB)) 
                # Используем psm 10 (один символ) как последнюю попытку
                text = pytesseract.image_to_string(large_pil, config='--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789').strip()
                text = re.sub(r'\D','', text)
                if text:
                    num = int(text)
                    if 1 <= num <= 31:
                        logger.info(f"Success on final Tesseract fallback (psm 10): {num}")
                        return num
            except Exception:
                pass

        logger.warning("No valid day number found by ensemble extractor")
        return None
    except Exception as e:
        logger.error(f"Error in _extract_day_ensemble: {e}")
        return None

# merge rects
def _merge_rects(rects: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    if not rects:
        return []
    rects_sorted = sorted(rects, key=lambda r: r[0])
    merged = []
    cur = rects_sorted[0]
    for r in rects_sorted[1:]:
        x1,y1,w1,h1 = cur
        x2,y2,w2,h2 = r
        if x2 <= x1 + w1 + max(4, int(0.1*w1)):
            nx = min(x1, x2); ny = min(y1, y2)
            nw = max(x1+w1, x2+w2) - nx
            nh = max(y1+h1, y2+h2) - ny
            cur = (nx, ny, nw, nh)
        else:
            merged.append(cur); cur = r
    merged.append(cur)
    return merged

# time slots detection
def _find_time_slots(img: np.ndarray) -> List[Tuple[str,str]]:
    try:
        h = img.shape[0]
        bottom = img[h//4:, :].copy()
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        text = ""
        if _PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(gray, config='-c tessedit_char_whitelist=0123456789:.- –— ')
            except Exception:
                text = ""
        if not text:
            return []
        text = text.replace('О','0').replace('о','0').replace('З','3').replace('з','3')
        slots = []
        for pattern in TIME_RANGE_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    h1,m1,h2,m2 = match
                    h1,m1,h2,m2 = int(h1), int(m1), int(h2), int(m2)
                    if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                        if h2 > h1 or (h2 == h1 and m2 > m1):
                            slots.append((f"{h1:02d}:{m1:02d}", f"{h2:02d}:{m2:02d}"))
                except Exception:
                    continue
        uniq = []
        seen = set()
        for s in slots:
            if s not in seen:
                seen.add(s); uniq.append(s)
        return uniq
    except Exception as e:
        logger.error(f"Error in _find_time_slots: {e}")
        return []

# -------------------
# Main parsers
# -------------------
class NewFormatSlotParser:
    """Parser for screenshots with yellow highlight day."""
    def __init__(self, debug: bool = False, allow_fallback: bool = False):
        self.debug = debug
        self.allow_fallback = allow_fallback
        if debug:
            logger.setLevel(logging.DEBUG)

    def process_image(self, image_input: Any) -> List[Dict]:
        img = _read_image(image_input)
        if img is None:
            logger.error("Failed to read image")
            return []
        logger.info(f"Processing image, shape: {img.shape}")

        yellow_box = _find_yellow_box(img)
        if yellow_box is None:
            logger.error("Yellow box not found")
            return []

        day = _extract_day_ensemble(img, yellow_box, debug=self.debug)
        if day is None:
            logger.error("Could not extract day from yellow box")
            if self.allow_fallback:
                day = datetime.now().day
                logger.warning(f"Using fallback day: {day}")
            else:
                logger.warning("Skipping image because day not determined (allow_fallback=False)")
                return []  # don't return slots with wrong date

        try:
            slot_date = date(FIXED_YEAR, FIXED_MONTH, day)
            iso_date = slot_date.isoformat()
            logger.info(f"Date for slots: {iso_date}")
        except Exception as e:
            logger.error(f"Invalid date: {e}")
            return []

        time_slots = _find_time_slots(img)
        if not time_slots:
            logger.warning(f"No time slots found for {iso_date}")
            return []

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

class MemorySlotParser:
    def __init__(self, debug: bool = False, allow_fallback: bool = False):
        self.debug = debug
        self.parser = NewFormatSlotParser(debug=debug, allow_fallback=allow_fallback)
        self.accumulated_slots = []
        self.screenshots_count = 0
        self.last_error = None

    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        try:
            logger.info(f"Processing screenshot #{self.screenshots_count+1}, is_last={is_last}")
            slots = self.parser.process_image(image_bytes)
            if slots:
                self.accumulated_slots.extend(slots)
            self.screenshots_count += 1
            if is_last:
                return self.get_all_slots()
            return slots
        except Exception as e:
            self.last_error = str(e); logger.error(f"Error: {e}"); return []

    def process_screenshot(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        return self.process_screenshot_from_memory(image_bytes, is_last)

    def process_image(self, image_input: Any) -> List[Dict]:
        try:
            return self.parser.process_image(image_input)
        except Exception as e:
            self.last_error = str(e); logger.error(f"Error: {e}"); return []

    def process_image_bytes(self, image_bytes: bytes) -> List[Dict]:
        try:
            return self.parser.process_image(image_bytes)
        except Exception as e:
            self.last_error = str(e); logger.error(f"Error: {e}"); return []

    def get_all_slots(self) -> List[Dict]:
        if not self.accumulated_slots:
            logger.warning("No accumulated slots to return"); return []
        seen = set(); unique = []
        for slot in self.accumulated_slots:
            key = (slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key); unique.append(slot)
        unique.sort(key=lambda s:(s["date"], s["startTime"])); logger.info(f"Returning {len(unique)} unique slots")
        return unique

    def clear(self):
        self.accumulated_slots=[]; self.screenshots_count=0; self.last_error=None

class SlotParser:
    def __init__(self, base_path: str, debug: bool = False, allow_fallback: bool = False):
        self.base_path = base_path
        self.parser = NewFormatSlotParser(debug=debug, allow_fallback=allow_fallback)
        self.cancelled_count = 0

    def process_all_screenshots(self) -> List[Dict]:
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}"); return []
        files = [os.path.join(self.base_path, f) for f in os.listdir(self.base_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if not files: logger.warning(f"No images found in: {self.base_path}"); return []
        files.sort(key=lambda x: os.path.getctime(x))
        logger.info(f"Found {len(files)} images")
        all_slots = []
        for i,filepath in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {os.path.basename(filepath)}")
            try:
                slots = self.parser.process_image(filepath)
                if slots: all_slots.extend(slots)
            except Exception as e:
                logger.error(f"Error: {e}")
        seen=set(); unique=[]
        for slot in all_slots:
            key=(slot["date"], slot["startTime"], slot["endTime"])
            if key not in seen:
                seen.add(key); unique.append(slot)
        unique.sort(key=lambda s:(s["date"], s["startTime"]))
        logger.info(f"Total: {len(unique)} unique slots")
        return unique

__all__ = ["SlotParser", "MemorySlotParser", "NewFormatSlotParser"]
