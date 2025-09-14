# ocr_module.py
# -*- coding: utf-8 -*-
"""
Упрощенный OCR парсер для извлечения слотов из скриншотов.
Возвращает только необходимые данные для API.
"""

import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract

# Путь к tesseract (настройте под вашу систему)
# Windows:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Linux/Mac (обычно в PATH):
# pytesseract.pytesseract.tesseract_cmd = "tesseract"


class SlotParser:
    def __init__(self, base_path: str):
        self.base_path = base_path
        
        # Месяцы для парсинга дат
        self.months = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        
        # Регулярные выражения
        months_alt = "|".join(self.months.keys())
        weekdays_alt = "понедельник|вторник|среда|четверг|пятница|суббота|воскресенье"
        
        # Заголовок даты: "25 августа, понедельник"
        self.date_heading_re = re.compile(
            rf"^\s*(\d{{1,2}})\s+({months_alt})\s*,\s*({weekdays_alt})\s*$",
            flags=re.IGNORECASE
        )
        
        # Диапазон недели (игнорируем)
        self.week_range_re = re.compile(
            rf"\d{{1,2}}\s+({months_alt})\s+\d{{4}}\s*[-–—]\s*\d{{1,2}}\s+({months_alt})\s+\d{{4}}",
            flags=re.IGNORECASE
        )
        
        # Поиск времени
        self.time_token_re = re.compile(r"(\d{1,2}[:.]\d{2})")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Предобработка изображения для улучшения OCR"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Увеличение и улучшение контраста
            scale = 1.4
            resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(resized)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Инвертируем если фон темный
            if np.mean(bw) < 127:
                bw = cv2.bitwise_not(bw)
                
            return bw
        except Exception as e:
            print(f"[preprocess_image] Ошибка: {e}")
            return None

    def extract_lines_with_coords(self, image_path: str) -> List[Dict]:
        """Извлечение текстовых строк с координатами"""
        # Предобработка
        processed = self.preprocess_image(image_path)
        if processed is None:
            return []
        
        try:
            # OCR
            pil_image = Image.fromarray(processed)
            data = pytesseract.image_to_data(pil_image, lang="rus", output_type=pytesseract.Output.DICT)
            
            # Группировка по строкам
            groups = {}
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                groups.setdefault(key, []).append(i)
            
            # Сборка строк
            lines = []
            for key, idxs in groups.items():
                idxs.sort(key=lambda j: data["left"][j])
                parts = [data["text"][j].strip() for j in idxs if data["text"][j].strip()]
                if not parts:
                    continue
                text = " ".join(parts)
                top = min(data["top"][j] for j in idxs)
                left = min(data["left"][j] for j in idxs)
                lines.append({"text": text, "y": int(top), "x": int(left)})
            
            lines.sort(key=lambda l: (l["y"], l["x"]))
            return lines
            
        except Exception as e:
            print(f"[extract_lines] OCR ошибка: {e}")
            return []

    def _pick_year_from_week_range(self, lines: List[Dict]) -> Optional[int]:
        """Извлечение года из диапазона недели"""
        for ln in lines[:6]:
            m = re.search(r"\b(\d{4})\b", ln["text"])
            if m:
                try:
                    return int(m.group(1))
                except:
                    pass
        return None

    def parse_date_heading(self, text: str, fallback_year: Optional[int]) -> Optional[Dict]:
        """Парсинг заголовка даты"""
        m = self.date_heading_re.match(text.strip())
        if not m:
            return None
            
        day = int(m.group(1))
        month_name = m.group(2).lower()
        month = self.months.get(month_name)
        year = fallback_year or datetime.now().year
        
        try:
            date_obj = datetime(year, month, day)
            return {"date": date_obj.strftime("%Y-%m-%d"), "y": None, "line_idx": None}
        except:
            return None

    def parse_time_in_text(self, text: str) -> Optional[Tuple[str, str]]:
        """Поиск пары времени в тексте"""
        tokens = self.time_token_re.findall(text)
        if len(tokens) < 2:
            return None
            
        # Нормализация
        t0 = tokens[0].replace(".", ":")
        t1 = tokens[1].replace(".", ":")
        
        try:
            datetime.strptime(t0, "%H:%M")
            datetime.strptime(t1, "%H:%M")
            return t0, t1
        except:
            return None

    def find_time_by_combination(self, lines: List[Dict], idx: int, max_window: int = 3, 
                                band_top: int = -10**9, band_bottom: int = 10**9) -> Optional[Tuple[int, str, str]]:
        """Поиск времени путем комбинирования соседних строк"""
        n = len(lines)
        
        # Комбинируем текущую строку с последующими
        for w in range(1, max_window + 1):
            j = idx + w - 1
            if j >= n:
                break
            if lines[j]["y"] < band_top or lines[j]["y"] > band_bottom:
                break
                
            combined = " ".join(lines[k]["text"] for k in range(idx, j + 1))
            res = self.parse_time_in_text(combined)
            if res:
                return (idx, res[0], res[1])
        
        # Пробуем с предыдущей строкой
        if idx - 1 >= 0 and lines[idx - 1]["y"] >= band_top and lines[idx - 1]["y"] <= band_bottom:
            combined = lines[idx - 1]["text"] + " " + lines[idx]["text"]
            res = self.parse_time_in_text(combined)
            if res:
                return (idx - 1, res[0], res[1])
                
        return None

    def parse_screenshot(self, lines: List[Dict], is_last_screenshot: bool = False) -> List[Dict]:
        """Парсинг одного скриншота"""
        if not lines:
            return []
            
        year_guess = self._pick_year_from_week_range(lines)
        
        # Находим заголовки дат
        date_blocks = []
        for idx, ln in enumerate(lines):
            if self.week_range_re.search(ln["text"]):
                continue
            di = self.parse_date_heading(ln["text"], fallback_year=year_guess)
            if di:
                di["y"] = ln["y"]
                di["line_idx"] = idx
                date_blocks.append(di)
                
        if not date_blocks:
            return []
            
        date_blocks.sort(key=lambda d: d["y"])
        slots = []
        seen_keys = set()
        
        if not is_last_screenshot:
            # Обрабатываем только первую дату
            first = date_blocks[0]
            first_y = first["y"]
            first_idx = first["line_idx"]
            second_y = date_blocks[1]["y"] if len(date_blocks) >= 2 else 10**9
            second_idx = date_blocks[1]["line_idx"] if len(date_blocks) >= 2 else len(lines) + 1
            
            for i in range(first_idx + 1, min(second_idx, len(lines))):
                ln = lines[i]
                if ln["y"] <= first_y or ln["y"] >= second_y:
                    continue
                    
                found = self.find_time_by_combination(lines, i, max_window=3, 
                                                    band_top=first_y, band_bottom=second_y)
                if not found:
                    continue
                    
                _, start, end = found
                slot = {
                    "date": first["date"],
                    "start": start,
                    "end": end
                }
                
                key = (slot["date"], slot["start"], slot["end"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    slots.append(slot)
        else:
            # Последний скриншот - обрабатываем все даты
            bands = []
            for i, db in enumerate(date_blocks):
                top_y = db["y"]
                bottom_y = date_blocks[i + 1]["y"] if i + 1 < len(date_blocks) else 10**9
                bands.append((db, top_y, bottom_y))
            
            for i, ln in enumerate(lines):
                if self.week_range_re.search(ln["text"]) or self.date_heading_re.match(ln["text"].strip()):
                    continue
                    
                y = ln["y"]
                assigned = None
                for db, top_y, bottom_y in bands:
                    if y > top_y and y < bottom_y:
                        assigned = (db, top_y, bottom_y)
                        break
                        
                if not assigned:
                    continue
                    
                db, top_y, bottom_y = assigned
                found = self.find_time_by_combination(lines, i, max_window=3, 
                                                    band_top=top_y, band_bottom=bottom_y)
                if not found:
                    continue
                    
                _, start, end = found
                slot = {
                    "date": db["date"],
                    "start": start,
                    "end": end
                }
                
                key = (slot["date"], slot["start"], slot["end"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    slots.append(slot)
                    
        return slots

    def process_all_screenshots(self) -> List[Dict]:
        """Обработка всех скриншотов в папке пользователя"""
        all_slots = []
        
        # Ищем все изображения в папке
        image_files = []
        for file in os.listdir(self.base_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(self.base_path, file))
        
        if not image_files:
            print(f"⚠️ Не найдено изображений в {self.base_path}")
            return []
        
        # Сортируем файлы по имени
        image_files.sort()
        
        # Обрабатываем каждый файл
        for idx, fp in enumerate(image_files):
            print(f"📄 Обработка {os.path.basename(fp)}")
            lines = self.extract_lines_with_coords(fp)
            print(f"   Извлечено строк: {len(lines)}")
            
            is_last = (idx == len(image_files) - 1)
            slots = self.parse_screenshot(lines, is_last_screenshot=is_last)
            print(f"   Найдено слотов: {len(slots)}")
            all_slots.extend(slots)
        
        # Сортировка и форматирование для API
        all_slots.sort(key=lambda s: (s["date"], s["start"]))
        
        # Преобразуем в формат API
        api_slots = []
        for slot in all_slots:
            api_slots.append({
                "date": slot["date"],
                "startTime": slot["start"],
                "endTime": slot["end"],
                "assignToSelf": True
            })
        
        return api_slots