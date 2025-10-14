# ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR-модуль для извлечения слотов из скриншотов.
Версия: HYBRID (PaddleOCR + template-matching classifier + template matching fallback).

Сохраняет прежний API и структуру (NewFormatSlotParser, MemorySlotParser, SlotParser).
"""

import os
import re
import logging
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO
from collections import Counter, defaultdict
import math

import cv2
import numpy as np
from PIL import Image

# Try to import PaddleOCR (preferred). If not available, we'll fallback to pytesseract.
try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except Exception:
    PaddleOCR = None
    _PADDLE_AVAILABLE = False

# pytesseract fallback
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

# --------- OCR Reader init ---------
def _init_paddle_reader():
    if not _PADDLE_AVAILABLE:
        logger.warning("PaddleOCR not available. Install with `pip install paddleocr paddlepaddle` for best results.")
        return None
    try:
        # cpu-friendly initialization; disable angle classifier for speed (we mostly have digits)
        reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        logger.info("PaddleOCR reader initialized.")
        return reader
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}")
        return None

_PADDLE_READER = _init_paddle_reader()

# --------- Вспомогательные функции ---------
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
    """Find the yellow square - same logic as original"""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 50, 50])
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
            if area < 500:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            ar = w / float(h)
            if 0.7 <= ar <= 1.5:
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
# Template-based digit templates and classifier (lightweight, no ML libs)
# ---------------------------
def _make_digit_templates(sizes=(28,36,48)):
    """
    Create edge templates for digits 0-9 for different sizes.
    Returns dict: {size: {digit: tpl_edge}}
    """
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

_DIGIT_TEMPLATES = _make_digit_templates((28,36,48))

def _template_classify_digit(digit_img_gray: np.ndarray) -> Tuple[int, float]:
    """
    Classify a small gray image that (likely) contains a single digit using templates.
    Returns (digit, score 0..100).
    """
    try:
        # preprocess: threshold + edges
        # Resize digit to multiple template sizes and compute best normalized cross-correlation on edges
        edges = cv2.Canny(digit_img_gray, 50, 150)
        best_digit = -1
        best_score = 0.0
        h0, w0 = edges.shape[:2]
        for size, tpl_dict in _DIGIT_TEMPLATES.items():
            # Resize edges to template scale (keep aspect by padding)
            try:
                # center-crop/pad to square then resize
                maxi = max(h0, w0)
                canvas = np.ones((maxi, maxi), dtype=np.uint8) * 0
                # place edges centered on canvas
                y_off = (maxi - h0)//2
                x_off = (maxi - w0)//2
                canvas[y_off:y_off+h0, x_off:x_off+w0] = edges
                resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC)
            except Exception:
                continue
            for d, tpl in tpl_dict.items():
                # compare tpl and resized via normalized correlation of binary images
                # convert to float and compute correlation
                tpl_f = tpl.astype(np.float32) / 255.0
                r_f = resized.astype(np.float32) / 255.0
                # compute similarity = 1 - normalized L2 distance (bounded)
                dist = np.linalg.norm(tpl_f - r_f)
                # normalize dist to [0, sqrt(size*size)]
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
        logger.debug(f"Template classifier error: {e}")
        return (-1, 0.0)

# ---------------------------
# Template matching on ROI edges to find digit positions
# ---------------------------
def _match_templates_on_roi_edges(roi_gray: np.ndarray, threshold_high=0.58, threshold_low=0.45):
    """
    Run templatematch for each digit template on ROI edges.
    Returns list of matches: {"digit":int,"x":int,"y":int,"score":float}
    """
    edges = cv2.Canny(roi_gray, 50, 150)
    matches = []
    for size, tpl_dict in _DIGIT_TEMPLATES.items():
        for d, tpl_edge in tpl_dict.items():
            th, tw = tpl_edge.shape
            if edges.shape[0] < th or edges.shape[1] < tw:
                continue
            try:
                res = cv2.matchTemplate(edges, tpl_edge, cv2.TM_CCOEFF_NORMED)
            except Exception:
                continue
            # simple peak finding (NMS)
            res_cp = res.copy()
            while True:
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res_cp)
                if maxVal < threshold_high:
                    break
                mx, my = maxLoc
                matches.append({"digit": d, "x": int(mx), "y": int(my), "score": float(maxVal)*100.0, "src":"tmpl"})
                # zero out neighborhood
                x0, y0 = mx, my
                x1 = max(0, x0 - tw//2); y1 = max(0, y0 - th//2)
                x2 = min(res_cp.shape[1], x0 + tw//2); y2 = min(res_cp.shape[0], y0 + th//2)
                res_cp[y1:y2, x1:x2] = 0
            # if no high matches, try lower threshold to collect more
            if not matches:
                res_cp = res.copy()
                while True:
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res_cp)
                    if maxVal < threshold_low:
                        break
                    mx, my = maxLoc
                    matches.append({"digit": d, "x": int(mx), "y": int(my), "score": float(maxVal)*100.0, "src":"tmpl"})
                    x0, y0 = mx, my
                    x1 = max(0, x0 - tw//2); y1 = max(0, y0 - th//2)
                    x2 = min(res_cp.shape[1], x0 + tw//2); y2 = min(res_cp.shape[0], y0 + th//2)
                    res_cp[y1:y2, x1:x2] = 0
    # return matches sorted by x
    matches.sort(key=lambda m: m["x"])
    return matches

# ---------------------------
# PaddleOCR wrapper and fallback to pytesseract
# ---------------------------
def _ocr_read_text_on_image(img_gray: np.ndarray) -> List[Tuple[str, float, Tuple[int,int,int,int]]]:
    """
    Returns list of (text, conf(0..100), bbox_in_img_coords) using PaddleOCR or pytesseract fallback.
    bbox format: (x_min, y_min, x_max, y_max)
    """
    results = []
    # Prefer paddle
    if _PADDLE_READER is not None:
        try:
            # PaddleOCR expects RGB image
            pil_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB) if len(img_gray.shape)==2 else cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
            # paddleocr returns list of lists; each entry: [[(x1,y1),(x2,y2),(x3,y3),(x4,y4)], (text, score)]
            raw = _PADDLE_READER.ocr(pil_rgb, cls=False)
            for line in raw:
                if not line:
                    continue
                bbox = line[0]
                text = line[1][0] if len(line[1])>0 else ""
                conf = float(line[1][1]) if len(line[1])>1 and isinstance(line[1][1], (float,int)) else 0.0
                xs = [int(p[0]) for p in bbox]; ys = [int(p[1]) for p in bbox]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                results.append((text, conf*100.0, (x_min, y_min, x_max, y_max)))
            return results
        except Exception as e:
            logger.debug(f"PaddleOCR failed: {e}")
            # fallback to pytesseract below

    # fallback: pytesseract (if available)
    if _PYTESSERACT_AVAILABLE:
        try:
            # pytesseract returns string; to get bbox we can use image_to_data
            data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT, config='-c tessedit_char_whitelist=0123456789/: ')
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
                results.append((txt, conf, (x,y,x+w,y+h)))
            return results
        except Exception as e:
            logger.debug(f"pytesseract failed: {e}")
            return []
    # if none available, return empty
    logger.warning("No OCR engine available (PaddleOCR and pytesseract missing).")
    return []

# ---------------------------
# Main: extract day hybrid
# ---------------------------
def _extract_day_hybrid(img: np.ndarray, box: Tuple[int,int,int,int]) -> Optional[int]:
    """
    Hybrid approach:
      - Crop exact ROI (small pad)
      - Run PaddleOCR (multiple scales) to get textual candidates
      - Run template matching across ROI to get digit positions/matches
      - Run template classifier for small digit crops (per candidate)
      - Aggregate candidates with weights: OCR weight=1.0, classifier weight=0.8, template weight=0.5
      - Apply validation rules for two-digit numbers (require OCR or classifier support)
    """
    try:
        x,y,w,h = box
        pad = max(0, int(min(w,h) * 0.05))
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        roi_bgr = img[y1:y2, x1:x2].copy()
        if roi_bgr.size == 0:
            logger.warning("Empty ROI for _extract_day_hybrid")
            return None
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # ---------- 1) OCR candidates (multiple scales) ----------
        ocr_tokens = []  # {"num":int, "conf":float, "x":int, "src":"ocr"}
        scales = [1.5, 2.5, 3.5]
        for s in scales:
            try:
                H0, W0 = roi_gray.shape
                img_r = cv2.resize(roi_gray, (int(W0*s), int(H0*s)), interpolation=cv2.INTER_CUBIC)
            except Exception:
                img_r = roi_gray.copy()
            ocr_results = _ocr_read_text_on_image(img_r)
            for text, conf, bbox in ocr_results:
                if not text:
                    continue
                tnorm = text.replace(' ', '').replace('O','0').replace('o','0').replace('l','1').replace('I','1')
                # if there's slash e.g., "4/12", take part before slash
                left = tnorm.split('/')[0] if '/' in tnorm else tnorm
                matches = re.findall(r'\d{1,2}', left)
                # bbox is in scaled coords -> map left x back
                if bbox:
                    bxmin = bbox[0]
                    x_coord = int(bxmin / s)
                else:
                    x_coord = None
                for m in matches:
                    v = int(m)
                    if 1 <= v <= 31:
                        ocr_tokens.append({"num":v, "conf": float(conf), "x": x_coord, "src":"ocr"})

        # ---------- 2) Template matching across ROI edges ----------
        template_matches = _match_templates_on_roi_edges(roi_gray)
        # convert to unified format tokens (template matches)
        tmpl_tokens = []
        for tm in template_matches:
            # tm already has 'digit', 'x', 'score'
            tmpl_tokens.append({"num": int(tm["digit"]), "conf": float(tm["score"]), "x": int(tm["x"]), "src":"tmpl"})

        # ---------- 3) Classifier predictions for candidate digit crops ----------
        # We'll attempt to crop around template matches and classify using template-classifier
        classifier_tokens = []
        # Use template matches as seeds for crops; if none, try to segment ROI into connected components
        seeds = template_matches if template_matches else []

        if seeds:
            for s in seeds:
                try:
                    # crop region around x position: width ~ w*0.35 or template size
                    cx = s["x"]
                    H, W = roi_gray.shape
                    crop_w = max(20, int(min(W, max(20, W*0.3))))
                    x0 = max(0, cx - crop_w//2)
                    x1c = min(W, cx + crop_w//2)
                    digit_crop = roi_gray[:, x0:x1c]
                    dnum, dscore = _template_classify_digit(digit_crop)
                    if dnum >= 0:
                        # store approx x (left)
                        classifier_tokens.append({"num": dnum, "conf": dscore, "x": x0, "src":"cls"})
                except Exception:
                    continue
        else:
            # fallback segmentation by contours (binarize and find bounding rects)
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

        # ---------- Merge tokens ----------
        merged = []
        merged.extend(ocr_tokens)
        merged.extend(tmpl_tokens)
        merged.extend(classifier_tokens)

        if not merged:
            logger.warning("No tokens from OCR/template/classifier in ROI.")
            return None

        # ---------- Build single-digit contributions ----------
        contributions = defaultdict(list)  # num -> list of contributions {conf, src, x}
        for t in merged:
            n = int(t["num"])
            contributions[n].append({"conf": float(t["conf"]), "src": t.get("src","unk"), "x": t.get("x")})

        # ---------- Build two-digit candidates from spatial ordering ----------
        tokens_with_x = [t for t in merged if t.get("x") is not None]
        if tokens_with_x:
            tokens_with_x.sort(key=lambda t: t["x"])
            roi_w = roi_gray.shape[1]
            # combine neighbors
            for i in range(len(tokens_with_x)):
                left = tokens_with_x[i]
                for j in (i+1, i+2):
                    if j >= len(tokens_with_x):
                        break
                    right = tokens_with_x[j]
                    dist = right["x"] - left["x"]
                    # require reasonable proximity
                    if dist < max(min(roi_w * 0.5, 80), 40):
                        num = left["num"]*10 + right["num"]
                        if 1 <= num <= 31:
                            contributions[num].append({"conf": float(left["conf"]), "src": left.get("src","unk"), "x": left.get("x")})
                            contributions[num].append({"conf": float(right["conf"]), "src": right.get("src","unk"), "x": right.get("x")})

        # ---------- Scoring candidates ----------
        # weights
        WEIGHTS = {"ocr": 1.0, "cls": 0.85, "tmpl": 0.6, "unk": 0.5}
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
            count = len(contribs)
            avg_conf = (total_conf / count) if count>0 else 0.0
            span = (max(xs) - min(xs)) if xs else 0
            # spatial penalty for too wide spans
            span_penalty = 1.0
            if span > max(roi_gray.shape[1] * 0.6, 120):
                span_penalty = 0.5
            # validity rule for two-digit
            is_two = (10 <= num <= 31)
            valid_two = True
            if is_two:
                # require at least one OCR or classifier evidence
                if (count_ocr + count_cls) >= 1:
                    valid_two = True
                else:
                    # if only template matches, require both parts high conf
                    # attempt to split by xs and check confs
                    if len(xs) >= 2:
                        # sort contribs by x and pick two highest-conf positions
                        contribs_spatial = sorted([c for c in contribs if c.get("x") is not None], key=lambda c:c["x"])
                        left_conf = contribs_spatial[0]["conf"] if len(contribs_spatial)>0 else 0
                        right_conf = contribs_spatial[1]["conf"] if len(contribs_spatial)>1 else 0
                        if not (left_conf >= 80 and right_conf >= 80):
                            valid_two = False
                    else:
                        valid_two = False
            final_score = total_weighted * span_penalty
            candidate_metrics[num] = {
                "final_score": final_score,
                "avg_conf": avg_conf,
                "count": count,
                "count_ocr": count_ocr,
                "count_cls": count_cls,
                "span": span,
                "valid_two": valid_two
            }

        # ---------- Choose best candidate ----------
        if not candidate_metrics:
            return None

        # prefer valid two-digit numbers, then by final_score, then avg_conf, then numeric desc
        sortable = []
        for num, m in candidate_metrics.items():
            two_valid_flag = 1 if (10 <= num <= 31 and m["valid_two"]) else 0
            # if two-digit but invalid, penalize the score
            score_val = m["final_score"]
            if 10 <= num <= 31 and not m["valid_two"]:
                score_val *= 0.45
            sortable.append((two_valid_flag, score_val, m["avg_conf"], num))
        sortable.sort(key=lambda t: (-t[0], -t[1], -t[2], -t[3]))

        chosen = int(sortable[0][3])
        logger.info(f"Day candidates (agg): { {k: candidate_metrics[k] for k in sorted(candidate_metrics.keys())} }, selected: {chosen}")
        return chosen

    except Exception as e:
        logger.error(f"Error extracting day (hybrid): {e}")
        return None

# ---------------------------
# Find time slots (uses PaddleOCR or pytesseract fallback)
# ---------------------------
def _find_time_slots(img: np.ndarray) -> List[Tuple[str,str]]:
    try:
        h = img.shape[0]
        bottom_part = img[h//4:, :].copy()
        gray = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2GRAY)
        # OCR
        raw = _ocr_read_text_on_image(gray)
        lines = []
        for text, conf, bbox in raw:
            if not text:
                continue
            t = text.replace('О','0').replace('о','0').replace('З','3').replace('з','3')
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
        # unique
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
# Parser classes (API preserved)
# ---------------------------
class NewFormatSlotParser:
    """Parser for screenshots with yellow day highlight."""
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

        day = _extract_day_hybrid(img, yellow_box)
        if day is None:
            logger.error("Could not extract day from yellow box")
            # fallback - current day
            day = datetime.now().day
            if day < 1 or day > 31:
                day = 1
            logger.warning(f"Using fallback day: {day}")

        # Build date
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
