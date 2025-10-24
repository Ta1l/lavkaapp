# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Новая детерминированная версия: строгий pipeline для извлечения числа в жёлтом квадратике.
Подход: многоступенчатая предобработка + template-classifier (edge templates) + агрегация.
Сохраняет прежний API и структуру.
"""

import os
import re
import logging
import math
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO
from collections import defaultdict, Counter

import cv2
import numpy as np
from PIL import Image

# Optional fallback OCR
try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    _PYTESSERACT_AVAILABLE = False

logger = logging.getLogger("lavka.ocr_module")
logger.setLevel(logging.INFO)

# --------- Constants ---------
FIXED_YEAR = 2025
FIXED_MONTH = 10

TIME_RANGE_PATTERNS = [
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    re.compile(r"(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})"),
]

# --------- Image IO helpers ---------
def _read_image(image_input: Any) -> Optional[np.ndarray]:
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

# --------- Find yellow box (same logic, tuned) ---------
def _find_yellow_box(img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Slightly robust yellow range
        lower = np.array([8, 60, 60])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        h = mask.shape[0]
        # calendar is at top; keep top half only
        mask[h//2:, :] = 0
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            ar = w / float(h) if h>0 else 0
            if 0.6 <= ar <= 1.6:
                valid.append((x,y,w,h,area))
        if not valid:
            return None
        valid.sort(key=lambda x: x[4], reverse=True)
        best = valid[0][:4]
        logger.info(f"Found yellow box at ({best[0]}, {best[1]}) size ({best[2]}x{best[3]})")
        return best
    except Exception as e:
        logger.error(f"Error finding yellow box: {e}")
        return None

# --------- Template digits creation (edge templates) ---------
def _make_templates(sizes=(28,36,48)):
    tpls = {}
    for size in sizes:
        digit_map = {}
        for d in range(10):
            canvas = np.ones((size, size), dtype=np.uint8) * 255
            font_scale = size / 40.0
            thickness = max(1, int(font_scale))
            ((tw, th), _) = cv2.getTextSize(str(d), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            org = ((size - tw)//2, (size + th)//2)
            cv2.putText(canvas, str(d), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,), thickness, cv2.LINE_AA)
            edge = cv2.Canny(canvas, 50, 150)
            digit_map[d] = edge
        tpls[size] = digit_map
    return tpls

_TEMPLATES = _make_templates((28,36,48))

# compute similarity between two binary edge images
def _edge_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return similarity in [0..1] based on normalized L2 distance."""
    try:
        a_f = a.astype(np.float32)/255.0
        b_f = b.astype(np.float32)/255.0
        dist = np.linalg.norm(a_f - b_f)
        max_norm = math.sqrt(a.shape[0]*a.shape[1]) + 1e-6
        sim = max(0.0, 1.0 - dist/max_norm)
        return sim
    except Exception:
        return 0.0

# Classify single digit patch using templates (search across sizes)
def _classify_digit_patch(patch_gray: np.ndarray) -> Tuple[int, float]:
    """Return (digit, score0..100). If not confident, return (-1, 0.0)."""
    try:
        # edges
        edges = cv2.Canny(patch_gray, 50, 150)
        h, w = edges.shape
        best_d = -1
        best_score = 0.0
        for size, digit_map in _TEMPLATES.items():
            # center on square, resize to size
            maxi = max(h,w,1)
            canvas = np.zeros((maxi, maxi), dtype=np.uint8)
            y_off = (maxi - h)//2
            x_off = (maxi - w)//2
            canvas[y_off:y_off+h, x_off:x_off+w] = edges
            try:
                resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)
            except Exception:
                continue
            for d, tpl in digit_map.items():
                s = _edge_similarity(resized, tpl)
                score = s * 100.0
                if score > best_score:
                    best_score = score
                    best_d = d
        if best_score < 40.0:
            # not confident
            return (-1, 0.0)
        return (best_d, best_score)
    except Exception as e:
        logger.debug(f"classify_digit_patch error: {e}")
        return (-1, 0.0)

# --------- NEW SIMPLIFIED extraction pipeline for day ---------
def _extract_day_deterministic(img: np.ndarray, box: Tuple[int,int,int,int], debug: bool=False) -> Optional[int]:
    """
    Simplified extraction that actually works:
    1. Extract ROI with small padding
    2. Convert to grayscale
    3. Upscale 4x for better OCR
    4. Apply Otsu threshold with inversion (white digits on black background)
    5. Small morphological cleanup
    6. Try OCR with different configs
    """
    try:
        x, y, w, h = box
        
        # Extract ROI with small padding
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        
        roi = img[y1:y2, x1:x2].copy()
        if roi.size == 0:
            logger.warning("Empty ROI")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # CRITICAL: Upscale for better OCR (4x works well)
        scale = 4
        gray_large = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply binary threshold with inversion (white digits on black background)
        _, binary = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Small morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Try OCR if available
        if not _PYTESSERACT_AVAILABLE:
            logger.warning("Tesseract not available, cannot extract day")
            return None
        
        # Try different Tesseract configurations
        configs = [
            '--psm 7 -c tessedit_char_whitelist=0123456789',   # Single text line
            '--psm 8 -c tessedit_char_whitelist=0123456789',   # Single word
            '--psm 13 -c tessedit_char_whitelist=0123456789',  # Raw line
            '--psm 6 -c tessedit_char_whitelist=0123456789',   # Uniform block
            '-c tessedit_char_whitelist=0123456789',           # Default
        ]
        
        results = []
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(binary_clean, config=config).strip()
                if text and text.isdigit():
                    num = int(text)
                    if 1 <= num <= 31:
                        results.append(num)
                        logger.debug(f"OCR found: {num} with config: {config}")
            except Exception as e:
                logger.debug(f"OCR error with config {config}: {e}")
        
        # Also try on original size (sometimes works better for small numbers)
        _, binary_small = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        for config in configs[:3]:  # Try first 3 configs on small version
            try:
                text = pytesseract.image_to_string(binary_small, config=config).strip()
                if text and text.isdigit():
                    num = int(text)
                    if 1 <= num <= 31:
                        results.append(num)
                        logger.debug(f"OCR found (small): {num}")
            except:
                pass
        
        # Additionally try with different threshold values
        for thresh_val in [100, 120, 140, 160]:
            _, binary_test = cv2.threshold(gray_large, thresh_val, 255, cv2.THRESH_BINARY_INV)
            try:
                text = pytesseract.image_to_string(binary_test, config='--psm 8 -c tessedit_char_whitelist=0123456789').strip()
                if text and text.isdigit():
                    num = int(text)
                    if 1 <= num <= 31:
                        results.append(num)
            except:
                pass
        
        # Analyze results - pick most common
        if results:
            from collections import Counter
            counter = Counter(results)
            most_common = counter.most_common(1)[0][0]
            logger.info(f"Day extraction results: {dict(counter)}, selected: {most_common}")
            return most_common
        
        logger.warning("No valid day number found by OCR")
        return None
        
    except Exception as e:
        logger.error(f"Error in _extract_day_deterministic: {e}")
        return None

# ---------------------------
# Helper: merge rects by overlap
# ---------------------------
def _merge_rects(rects: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
    if not rects:
        return []
    # naive merging: sort by x and merge overlapping/close rects
    rects_sorted = sorted(rects, key=lambda r: r[0])
    merged = []
    cur = rects_sorted[0]
    for r in rects_sorted[1:]:
        x1,y1,w1,h1 = cur
        x2,y2,w2,h2 = r
        # if intersect or close horizontally
        if x2 <= x1 + w1 + max(4, int(0.1*w1)):
            # merge
            nx = min(x1, x2)
            ny = min(y1, y2)
            nw = max(x1+w1, x2+w2) - nx
            nh = max(y1+h1, y2+h2) - ny
            cur = (nx, ny, nw, nh)
        else:
            merged.append(cur)
            cur = r
    merged.append(cur)
    return merged

# small wrapper for classifying using templates on raw grayscale patch
def _classify_digit_patch_from_gray(patch_gray: np.ndarray) -> Tuple[int, float]:
    """
    Pre-normalize patch then call _classify_digit_patch which expects grayscale patch.
    """
    try:
        # resize small patches if too tiny
        h,w = patch_gray.shape
        if h < 6 or w < 6:
            # upsample
            factor = max(2, int(6.0 / min(h,w)))
            patch_gray = cv2.resize(patch_gray, (w*factor, h*factor), interpolation=cv2.INTER_CUBIC)
        # apply small blur to reduce aliasing
        patch_gray = cv2.GaussianBlur(patch_gray, (3,3), 0)
        # Ensure contrast normalization
        patch_gray = cv2.normalize(patch_gray, None, 0, 255, cv2.NORM_MINMAX)
        return _classify_digit_patch(patch_gray)
    except Exception as e:
        logger.debug(f"_classify_digit_patch_from_gray error: {e}")
        return (-1, 0.0)

# ---------------------------
# Time slots detection - use pytesseract if available, else try rough heuristic
# ---------------------------
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
        # fallback naive recognition by looking for patterns in raw pixels via OCR-like heuristics is complex;
        # so if pytesseract not available we return empty
        if not text:
            return []
        # normalize some cyrillic-looking chars
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
        # uniq
        uniq = []
        seen = set()
        for s in slots:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq
    except Exception as e:
        logger.error(f"Error in _find_time_slots: {e}")
        return []

# ---------------------------
# Main parser classes (API preserved)
# ---------------------------
class NewFormatSlotParser:
    """Parser for screenshots with yellow highlight day."""
    def __init__(self, debug: bool = False):
        self.debug = debug
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

        # Use deterministic extractor
        day = _extract_day_deterministic(img, yellow_box, debug=self.debug)
        if day is None:
            logger.error("Could not extract day from yellow box")
            # fallback to today's day
            day = datetime.now().day
            if day < 1 or day > 31:
                day = 1
            logger.warning(f"Using fallback day: {day}")

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
    """Processor for screenshots from memory (backwards-compatible)."""
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
    """Parser for folder of screenshots."""
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