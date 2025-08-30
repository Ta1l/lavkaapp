# -*- coding: utf-8 -*-
"""
OCR модуль для распознавания слотов с расписания.
Использует pytesseract + opencv для обработки изображений.
Формат возвращаемых слотов:
{
  "date": "2025-08-31",
  "startTime": "09:00",
  "endTime": "17:00",
  "assignToSelf": true
}
"""

import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract


class SlotParser:
    def __init__(self, base_path: str = "./slots"):
        self.base_path = base_path

        # месяцы и дни недели
        self.months = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        self.weekdays = {
            "понедельник": 0, "вторник": 1, "среда": 2, "четверг": 3,
            "пятница": 4, "суббота": 5, "воскресенье": 6
        }

        months_alt = "|".join(self.months.keys())
        weekdays_alt = "|".join(self.weekdays.keys())

        # заголовок даты
        self.date_heading_re = re.compile(
            rf"^\s*(\d{{1,2}})\s+({months_alt})\s*,\s*({weekdays_alt})\s*$",
            flags=re.IGNORECASE
        )
        # поиск времени
        self.time_token_re = re.compile(r"(\d{1,2}[:.]\d{2})")

        # временная папка
        self.temp_dir = os.path.join(self.base_path, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    # ---------- предобработка ----------
    def preprocess_image(self, image_path: str) -> str:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(bw) < 127:
                bw = cv2.bitwise_not(bw)
            out_path = os.path.join(self.temp_dir, f"proc_{os.path.basename(image_path)}")
            cv2.imwrite(out_path, bw)
            return out_path
        except Exception as e:
            print(f"[preprocess_image] {e}")
            return image_path

    # ---------- OCR ----------
    def extract_lines(self, image_path: str):
        proc = self.preprocess_image(image_path)
        try:
            data = pytesseract.image_to_data(Image.open(proc), lang="rus", output_type=pytesseract.Output.DICT)
            groups = {}
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                groups.setdefault(key, []).append(i)
            lines = []
            for idxs in groups.values():
                idxs.sort(key=lambda j: data["left"][j])
                parts = [data["text"][j].strip() for j in idxs if data["text"][j].strip()]
                if not parts:
                    continue
                text = " ".join(parts)
                top = min(data["top"][j] for j in idxs)
                lines.append({"text": text, "y": int(top)})
            lines.sort(key=lambda l: l["y"])
            return lines
        except Exception as e:
            print(f"[extract_lines] OCR error: {e}")
            return []

    def parse_date_heading(self, text: str, fallback_year: Optional[int]) -> Optional[str]:
        m = self.date_heading_re.match(text.strip())
        if not m:
            return None
        day = int(m.group(1))
        month = self.months.get(m.group(2).lower())
        year = fallback_year or datetime.now().year
        try:
            date_obj = datetime(year, month, day)
            return date_obj.strftime("%Y-%m-%d")
        except:
            return None

    def parse_time_in_text(self, text: str) -> Optional[Tuple[str, str]]:
        tokens = self.time_token_re.findall(text)
        if len(tokens) < 2:
            return None
        t0 = tokens[0].replace(".", ":")
        t1 = tokens[1].replace(".", ":")
        try:
            datetime.strptime(t0, "%H:%M")
            datetime.strptime(t1, "%H:%M")
            return t0, t1
        except:
            return None

    # ---------- основной парсинг ----------
    def parse_screenshot(self, lines: List[Dict]) -> List[Dict]:
        slots: List[Dict] = []
        current_date = None
        for ln in lines:
            d = self.parse_date_heading(ln["text"], fallback_year=datetime.now().year)
            if d:
                current_date = d
                continue
            if not current_date:
                continue
            times = self.parse_time_in_text(ln["text"])
            if times:
                start, end = times
                slots.append({
                    "date": current_date,
                    "startTime": start,
                    "endTime": end,
                    "assignToSelf": True
                })
        return slots

    def process_all_screenshots(self) -> List[Dict]:
        all_slots: List[Dict] = []
        for fn in sorted(os.listdir(self.base_path)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fp = os.path.join(self.base_path, fn)
            lines = self.extract_lines(fp)
            slots = self.parse_screenshot(lines)
            if slots:
                print(f"📄 {fn}: найдено {len(slots)} слотов")
            else:
                print(f"❌ {fn}: слоты не найдены")
            all_slots.extend(slots)
        return all_slots
