# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
HYBRID версия: PaddleOCR (если доступен) + pytesseract (fallback) + template-based classifier & matching.
Сохраняет прежний API и структуру (NewFormatSlotParser, MemorySlotParser, SlotParser).
"""

import os
import re
import logging
import math
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

# Try to import PaddleOCR (preferred)
try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except Exception:
    PaddleOCR = None
    _PADDLE_AVAILABLE = False

# Try to import pytesseract (fallback)
try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    _PYTESSERACT_AVAILABLE = False

logger = logging.getLogger("lavka.ocr_module")
logger.setLevel(logging.INFO)

# --------- Константы ---------
FIXED_YEAR = 2025
FIXED_MONTH = 10

TIME_RANGE_PATTERNS = [
    re.compile(r"(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})"),
    re.compile(r"(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})"),
    re.compile(r"(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})"),
]

# --------- Initialize PaddleOCR safely (various versions) ---------
def _init_paddle_reader():
    if not _PADDLE_AVAILABLE:
        logger.info("PaddleOCR not installed; will use pytesseract/template fallbacks.")
        return None
    # try a couple init signatures to support different paddleocr versions
    try:
        reader = PaddleOCR(use_angle_cls=False, lang="en")
        logger.info("Initialized PaddleOCR (use_angle_cls=False, lang='en').")
        return reader
    except TypeError:
        # maybe older/newer signature
        try:
            reader = PaddleOCR(lang="en")
            logger.info("Initialized PaddleOCR (lang='en').")
            return reader
        except Exception as e:
            logger.warning(f"PaddleOCR init fallback failed: {e}")
            return None
    except Exception as e:
        logger.warning(f"PaddleOCR init failed: {e}")
        return None

_PADDLE_READER = _init_paddle_reader()

# --------- Helper functions (image IO, find yellow box) ---------
def _read_image(image_input: Any) -> Optional[np.ndarray]:
    """Read image from path/bytes/BytesIO/PIL.Image -> BGR numpy array"""
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

def _find_yellow_box(img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """Find the yellow square region (top half) and return (x,y,w,h) or None"""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Slightly broader yellow/orange range
        lower = np.array([8, 60, 60])
        upper = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        h = mask.shape[0]
        mask[h//2:, :] = 0  # search only top half
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:  # slightly smaller threshold
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            ar = w / float(h) if h>0 else 0
            if 0.6 <= ar <= 1.6:
                valid.append((x,y,w,h,area))
        if not valid:
            return None
        valid.sort(key=lambda b: b[4], reverse=True)
        best = valid[0][:4]
        logger.info(f"Found yellow box at ({best[0]}, {best[1]}) size ({best[2]}x{best[3]})")
        return best
    except Exception as e:
        logger.error(f"Error finding yellow box: {e}")
        return None

# ---------------------------
# Template digit creation & classifier (no ML libs required)
# ---------------------------
def _make_edge_templates(sizes=(28,36,48)):
    tpls = {}
    for size in sizes:
        tpl_dict = {}
        for d in range(10):
            canvas = np.ones((size, size), dtype=np.uint8) * 255
            font_scale = size / 40.0
            thickness = max(1, int(font_scale))
            ((tw, th), _) = cv2.getTextSize(str(d), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            org = ((size - tw)//2, (size + th)//2)
            cv2.putText(canvas, str(d), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,), thickness, cv2.LINE_AA)
            tpl_edge = cv2.Canny(canvas, 50, 150)
            tpl_dict[d] = tpl_edge
        tpls[size] = tpl_dict
    return tpls

_DIGIT_TEMPLATES = _make_edge_templates((28,36,48))

def _template_classify_digit(digit_img_gray: np.ndarray) -> Tuple[int, float]:
    """
    Classify a grayscale image (single digit crop) by matching to edge templates.
    Returns (digit, score 0..100). If uncertain returns (-1, 0.0).
    """
    try:
        edges = cv2.Canny(digit_img_gray, 50, 150)
        h0, w0 = edges.shape[:2]
        best_digit = -1
        best_score = 0.0
        for size, tpl_dict in _DIGIT_TEMPLATES.items():
            # normalize to square canvas
            maxi = max(h0, w0, 1)
            canvas = np.ones((maxi, maxi), dtype=np.uint8) * 0
            y_off = (maxi - h0)//2
            x_off = (maxi - w0)//2
            canvas[y_off:y_off+h0, x_off:x_off+w0] = edges
            try:
                resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)
            except Exception:
                continue
            for d, tpl in tpl_dict.items():
                # compute similarity (1 - normalized L2)
                tpl_f = tpl.astype(np.float32) / 255.0
                r_f = resized.astype(np.float32) / 255.0
                dist = np.linalg.norm(tpl_f - r_f)
                norm_max = math.sqrt(size*size)
                sim = max(0.0, 1.0 - (dist / (norm_max + 1e-6)))
                score = sim * 100.0
                if score > best_score:
                    best_score = score
                    best_digit = d
        if best_digit < 0:
            return (-1, 0.0)
        return (int(best_digit), float(best_score))
    except Exception as e:
        logger.debug(f"template classifier error: {e}")
        return (-1, 0.0)

def _nms_peaks(res, tpl_w, tpl_h, thresh=0.55):
    """Simple NMS for matchTemplate results, returns list of (x,y,score) peaks above thresh."""
    matches = []
    sm = res.copy()
    while True:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(sm)
        if maxVal < thresh:
            break
        mx, my = maxLoc
        matches.append((mx, my, float(maxVal)))
        # zero neighborhood
        x0, y0 = mx, my
        x1 = max(0, x0 - tpl_w//2); y1 = max(0, y0 - tpl_h//2)
        x2 = min(sm.shape[1], x0 + tpl_w//2); y2 = min(sm.shape[0], y0 + tpl_h//2)
        sm[y1:y2, x1:x2] = 0
    return matches

def _match_templates_on_roi_edges(roi_gray: np.ndarray, thr_high=0.58, thr_low=0.45):
    """Return list of matches dicts: {'digit', 'x', 'y', 'score', 'src'} sorted by x"""
    edges = cv2.Canny(roi_gray, 50, 150)
    matches_out = []
    for size, tpl_dict in _DIGIT_TEMPLATES.items():
        for d, tpl_edge in tpl_dict.items():
            th, tw = tpl_edge.shape
            if edges.shape[0] < th or edges.shape[1] < tw:
                continue
            try:
                res = cv2.matchTemplate(edges, tpl_edge, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            peaks = _nms_peaks(res, tpl_edge.shape[1], tpl_edge.shape[0], thresh=thr_high)
            if not peaks:
                peaks = _nms_peaks(res, tpl_edge.shape[1], tpl_edge.shape[0], thresh=thr_low)
            for mx, my, score in peaks:
                matches_out.append({"digit": int(d), "x": int(mx), "y": int(my), "score": float(score*100.0), "src":"tmpl"})
    matches_out.sort(key=lambda m: m["x"])
    return matches_out

# ---------------------------
# OCR wrapper: PaddleOCR preferred, pytesseract fallback
# ---------------------------
def _ocr_read_text_on_image(img: np.ndarray) -> List[Tuple[str, float, Tuple[int,int,int,int]]]:
    """
    Return list of (text, conf(0..100), bbox=(x_min,y_min,x_max,y_max)).
    Accepts BGR or grayscale image (numpy).
    """
    out = []
    try:
        # prepare RGB for paddle if necessary
        if _PADDLE_READER is not None:
            try:
                # paddleocr expects RGB image (H x W x 3) numpy
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape)==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                raw = _PADDLE_READER.ocr(img_rgb, cls=False)
                # raw format: list of [ [box_points], (text, score) ] OR nested; handle both
                for line in raw:
                    try:
                        box = line[0]
                        text = line[1][0] if isinstance(line[1], (list,tuple)) and len(line[1])>0 else (line[1] if isinstance(line[1], str) else "")
                        conf = float(line[1][1]) if isinstance(line[1], (list,tuple)) and len(line[1])>1 else 0.0
                        xs = [int(p[0]) for p in box]; ys = [int(p[1]) for p in box]
                        out.append((text, conf*100.0, (min(xs), min(ys), max(xs), max(ys))))
                    except Exception:
                        # Some versions return simpler tuples
                        try:
                            text = str(line[1])
                            conf = 0.0
                            out.append((text, conf, (0,0,img.shape[1], img.shape[0])))
                        except Exception:
                            continue
                return out
            except Exception as e:
                logger.debug(f"PaddleOCR read error, falling back: {e}")
        # pytesseract fallback
        if _PYTESSERACT_AVAILABLE:
            try:
                # get data with box info
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='-c tessedit_char_whitelist=0123456789/: ')
                n = len(data.get("text", []))
                for i in range(n):
                    txt = (data["text"][i] or "").strip()
                    if not txt:
                        continue
                    conf_raw = data.get("conf", [])[i]
                    try:
                        conf = float(conf_raw)
                    except Exception:
                        conf = -1.0
                    x = int(data.get("left", [0])[i])
                    y = int(data.get("top", [0])[i])
                    w = int(data.get("width", [0])[i])
                    h = int(data.get("height", [0])[i])
                    out.append((txt, conf, (x,y,x+w,y+h)))
                return out
            except Exception as e:
                logger.debug(f"pytesseract read error: {e}")
        # If no OCR engines, return empty
        logger.debug("No OCR engine available (PaddleOCR/pytesseract).")
        return out
    except Exception as e:
        logger.error(f"_ocr_read_text_on_image error: {e}")
        return out

# ---------------------------
# Hybrid day extraction
# ---------------------------
def _extract_day_hybrid(img: np.ndarray, box: Tuple[int,int,int,int]) -> Optional[int]:
    """
    Hybrid extraction: crop ROI, preprocess, get OCR tokens (multi-scale),
    get template matches, classify digit crops, aggregate with weights and rules.
    Returns integer day 1..31 or None.
    """
    try:
        x,y,w,h = box
        # crop with small padding
        pad = max(0, int(min(w,h)*0.05))
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(img.shape[1], x+w+pad), min(img.shape[0], y+h+pad)
        roi_bgr = img[y1:y2, x1:x2].copy()
        if roi_bgr.size == 0:
            return None

        # Preprocess ROI: LAB + CLAHE on L, then slight blur
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2,a,b))
        roi_bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        roi_gray = cv2.cvtColor(roi_bgr2, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.medianBlur(roi_gray, 3)

        # Multi-scale OCR candidates
        ocr_candidates = []  # dicts: {"num", "conf", "x", "src"}
        scales = [1.5, 2.5]  # avoid too big to speed up
        for s in scales:
            try:
                H0, W0 = roi_gray.shape
                img_s = cv2.resize(roi_gray, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_CUBIC)
            except Exception:
                img_s = roi_gray.copy()
            ocr_out = _ocr_read_text_on_image(img_s)
            for text, conf, bbox in ocr_out:
                if not text:
                    continue
                tnorm = str(text).replace(' ', '').replace('O','0').replace('o','0').replace('l','1').replace('I','1')
                left_part = tnorm.split('/')[0] if '/' in tnorm else tnorm
                nums = re.findall(r'\d{1,2}', left_part)
                # map bbox left coordinate back to ROI space if available
                x_coord = None
                if bbox:
                    bx = bbox[0]
                    try:
                        x_coord = int(bx / s)
                    except Exception:
                        x_coord = None
                for nstr in nums:
                    numv = int(nstr)
                    if 1 <= numv <= 31:
                        ocr_candidates.append({"num": numv, "conf": float(conf), "x": x_coord, "src":"ocr"})

        # Template matches
        tmpl_matches = _match_templates_on_roi_edges(roi_gray)
        tmpl_tokens = [{"num": m["digit"], "conf": m["score"], "x": m["x"], "src":"tmpl"} for m in tmpl_matches]

        # Classifier tokens (try to crop around template matches, else find contours)
        classifier_tokens = []
        if tmpl_matches:
            # use positions to create crops and classify
            for m in tmpl_matches:
                try:
                    cx = m["x"]
                    H, W = roi_gray.shape
                    # try cropping a box around expected digit region
                    wbox = max(12, int(W*0.25))
                    x0 = max(0, cx - wbox//2)
                    x1c = min(W, cx + wbox//2)
                    crop = roi_gray[:, x0:x1c]
                    if crop.size == 0:
                        continue
                    dnum, dscore = _template_classify_digit(crop)
                    if dnum >= 0:
                        classifier_tokens.append({"num": dnum, "conf": dscore, "x": x0, "src":"cls"})
                except Exception:
                    continue
        else:
            # fallback: detect connected components after binarization and classify
            try:
                _, thr = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    x0,y0,w0,h0 = cv2.boundingRect(cnt)
                    if w0 < 6 or h0 < 8:
                        continue
                    crop = roi_gray[y0:y0+h0, x0:x0+w0]
                    dnum, dscore = _template_classify_digit(crop)
                    if dnum >= 0:
                        classifier_tokens.append({"num": dnum, "conf": dscore, "x": x0, "src":"cls"})
            except Exception:
                pass

        # Merge tokens
        merged_tokens = []
        merged_tokens.extend(ocr_candidates)
        merged_tokens.extend(tmpl_tokens)
        merged_tokens.extend(classifier_tokens)

        if not merged_tokens:
            logger.warning("No tokens found in ROI by any method.")
            return None

        # Build contributions per single digit
        contributions = defaultdict(list)  # num -> list of contributions {conf, src, x}
        for t in merged_tokens:
            n = int(t["num"])
            contributions[n].append({"conf": float(t["conf"]), "src": t.get("src","unk"), "x": t.get("x")})

        # Try to build two-digit numbers by spatial ordering of tokens_with_x
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
                    if dist < max(min(roi_w * 0.5, 100), 35):
                        num = left["num"]*10 + right["num"]
                        if 1 <= num <= 31:
                            contributions[num].append({"conf": float(left["conf"]), "src": left.get("src","unk"), "x": left.get("x")})
                            contributions[num].append({"conf": float(right["conf"]), "src": right.get("src","unk"), "x": right.get("x")})

        # Scoring candidates
        WEIGHTS = {"ocr": 1.0, "cls": 0.9, "tmpl": 0.6, "unk": 0.5}
        candidate_metrics = {}
        for num, contribs in contributions.items():
            total_weighted = 0.0
            total_conf = 0.0
            xs = []
            count_ocr = 0
            count_cls = 0
            for c in contribs:
                src = c.get("src","unk")
                conf = float(c.get("conf", 0.0))
                wsrc = WEIGHTS.get(src, 0.5)
                total_weighted += conf * wsrc
                total_conf += conf
                if c.get("x") is not None:
                    xs.append(int(c["x"]))
                if src == "ocr":
                    count_ocr += 1
                if src == "cls":
                    count_cls += 1
            cnt = len(contribs)
            avg_conf = (total_conf / cnt) if cnt>0 else 0.0
            span = (max(xs) - min(xs)) if xs else 0
            span_pen = 1.0
            if span > max(roi_gray.shape[1]*0.6, 120):
                span_pen = 0.5
            is_two = (10 <= num <= 31)
            valid_two = True
            if is_two:
                if (count_ocr + count_cls) >= 1:
                    valid_two = True
                else:
                    if len(xs) >= 2:
                        sorted_cs = sorted([c for c in contribs if c.get("x") is not None], key=lambda c:c["x"])
                        left_conf = sorted_cs[0]["conf"] if len(sorted_cs)>0 else 0
                        right_conf = sorted_cs[1]["conf"] if len(sorted_cs)>1 else 0
                        if not (left_conf >= 80 and right_conf >= 80):
                            valid_two = False
                    else:
                        valid_two = False
            final_score = total_weighted * span_pen
            candidate_metrics[num] = {
                "final_score": final_score,
                "avg_conf": avg_conf,
                "count": cnt,
                "count_ocr": count_ocr,
                "count_cls": count_cls,
                "span": span,
                "valid_two": valid_two
            }

        if not candidate_metrics:
            return None

        # pick best: prefer valid two-digit, then final_score, then avg_conf, then numeric
        sortable = []
        for num, m in candidate_metrics.items():
            two_flag = 1 if (10 <= num <= 31 and m["valid_two"]) else 0
            score_val = m["final_score"]
            if 10 <= num <= 31 and not m["valid_two"]:
                score_val *= 0.45
            sortable.append((two_flag, score_val, m["avg_conf"], num))
        sortable.sort(key=lambda t: (-t[0], -t[1], -t[2], -t[3]))

        chosen_num = int(sortable[0][3])
        # detailed debug log if debug level
        logger.info(f"Day candidates (agg): { {k: candidate_metrics[k] for k in sorted(candidate_metrics.keys())} }, selected: {chosen_num}")
        return chosen_num

    except Exception as e:
        logger.error(f"Error in _extract_day_hybrid: {e}")
        return None

# ---------------------------
# Find time slots (PaddleOCR/pytesseract fallback)
# ---------------------------
def _find_time_slots(img: np.ndarray) -> List[Tuple[str,str]]:
    try:
        h = img.shape[0]
        bottom_part = img[h//4:, :].copy()
        gray = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2GRAY)
        raw = _ocr_read_text_on_image(gray)
        lines = []
        for text, conf, bbox in raw:
            if not text:
                continue
            t = str(text).replace('О','0').replace('о','0').replace('З','3').replace('з','3')
            lines.append(t)
        full_text = "\n".join(lines)
        slots = []
        for pattern in TIME_RANGE_PATTERNS:
            matches = pattern.findall(full_text)
            for match in matches:
                try:
                    h1,m1,h2,m2 = match
                    h1,m1,h2,m2 = int(h1), int(m1), int(h2), int(m2)
                    if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                        if h2 > h1 or (h2 == h1 and m2 > m1):
                            slots.append((f"{h1:02d}:{m1:02d}", f"{h2:02d}:{m2:02d}"))
                except Exception:
                    continue
        # deduplicate
        uniq = []
        seen = set()
        for s in slots:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq
    except Exception as e:
        logger.error(f"Error finding time slots: {e}")
        return []

# ---------------------------
# Main parser classes (API preserved)
# ---------------------------
class NewFormatSlotParser:
    """Парсер для скриншотов с выделением дня."""
    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

    def process_image(self, image_input: Any) -> List[Dict]:
        # 1. read image
        img = _read_image(image_input)
        if img is None:
            logger.error("Failed to read image")
            return []
        logger.info(f"Processing image, shape: {img.shape}")

        # 2. find yellow box
        yellow_box = _find_yellow_box(img)
        if yellow_box is None:
            logger.error("Yellow box not found")
            return []
        # 3. extract day (hybrid)
        day = _extract_day_hybrid(img, yellow_box)
        if day is None:
            logger.error("Could not extract day from yellow box")
            # fallback to current day
            day = datetime.now().day
            if day < 1 or day > 31:
                day = 1
            logger.warning(f"Using fallback day: {day}")

        # 4. build iso date
        try:
            slot_date = date(FIXED_YEAR, FIXED_MONTH, day)
            iso_date = slot_date.isoformat()
            logger.info(f"Date for slots: {iso_date}")
        except Exception as e:
            logger.error(f"Invalid date: {e}")
            return []

        # 5. find time slots
        time_slots = _find_time_slots(img)
        if not time_slots:
            logger.warning(f"No time slots found for {iso_date}")
            return []

        # 6. build result
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

    def process_screenshot(self, image_bytes: BytesIO, is_last: bool=False) -> List[Dict]:
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
