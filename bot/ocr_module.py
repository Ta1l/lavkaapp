# bot/ocr_module.py
# -*- coding: utf-8 -*-
"""
OCR парсер для извлечения слотов из скриншотов.
Адаптирован для работы на Linux/Ubuntu и Windows.
"""

import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytesseract

# Настраиваем логгер
logger = logging.getLogger("lavka.ocr_module")

# Путь к tesseract автоматически определяется по ОС
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # На Ubuntu/Linux tesseract обычно в PATH после установки
    pytesseract.pytesseract.tesseract_cmd = "tesseract"


class SlotParser:
    def __init__(self, base_path: str):
        """
        base_path: путь к папке со скриншотами пользователя
        """
        self.base_path = base_path
        
        # Месяцы для парсинга дат
        self.months = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
        }
        
        # Статусы для поиска
        self.status_map = {
            "выполнен с опозданием": "Выполнен с опозданием",
            "выполнен": "Выполнен",
            "отменен": "Отменён",
            "отменён": "Отменён",
            "отмен": "Отменён"
        }
        
        # Регулярные выражения
        months_alt = "|".join(self.months.keys())
        weekdays_alt = "понедельник|вторник|среда|четверг|пятница|суббота|воскресенье"
        
        # Заголовок даты: "25 августа, понедельник"
        self.date_heading_re = re.compile(
            rf"^\s*(\d{{1,2}})\s+({months_alt})\s*,\s*({weekdays_alt})\s*$",
            flags=re.IGNORECASE
        )
        
        # Поиск времени - улучшенный паттерн для формата "8:00 - 12:00"
        self.time_range_re = re.compile(r"(\d{1,2}[:.]\d{2})\s*[-–—]\s*(\d{1,2}[:.]\d{2})")
        self.time_token_re = re.compile(r"(\d{1,2}[:.]\d{2})")
        
        # Счетчик отмененных слотов для статистики
        self.cancelled_count = 0

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Предобработка изображения для улучшения OCR"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Cannot read image: {image_path}")
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
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None

    def extract_lines_with_coords(self, image_path: str) -> List[Dict]:
        """Извлечение текстовых строк с координатами"""
        processed = self.preprocess_image(image_path)
        if processed is None:
            # Попытка работать с оригинальным изображением если предобработка не удалась
            try:
                img = cv2.imread(image_path)
                if img is None:
                    return []
                processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                return []
        
        try:
            pil_image = Image.fromarray(processed)
            data = pytesseract.image_to_data(pil_image, lang="rus", output_type=pytesseract.Output.DICT)
            
            groups = {}
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                groups.setdefault(key, []).append(i)
            
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
            logger.error(f"Error extracting text from {image_path}: {e}")
            return []

    def parse_date_heading(self, text: str) -> Optional[Dict]:
        """Парсинг заголовка даты"""
        m = self.date_heading_re.match(text.strip())
        if not m:
            return None
            
        day = int(m.group(1))
        month_name = m.group(2).lower()
        month = self.months.get(month_name)
        
        # Используем текущий год всегда (слоты только на текущую неделю)
        year = datetime.now().year
        
        # Проверяем, не нужно ли использовать следующий год (для конца декабря)
        current_month = datetime.now().month
        if current_month == 12 and month == 1:
            year += 1
        
        try:
            date_obj = datetime(year, month, day)
            return {"date": date_obj.strftime("%Y-%m-%d"), "y": None, "line_idx": None}
        except:
            return None

    def parse_time_in_text(self, text: str) -> Optional[Tuple[str, str]]:
        """Поиск пары времени в тексте"""
        # Сначала пробуем найти полный формат "8:00 - 12:00"
        match = self.time_range_re.search(text)
        if match:
            t0 = match.group(1).replace(".", ":")
            t1 = match.group(2).replace(".", ":")
            try:
                datetime.strptime(t0, "%H:%M")
                datetime.strptime(t1, "%H:%M")
                return t0, t1
            except:
                pass
        
        # Если не нашли, ищем два отдельных времени
        tokens = self.time_token_re.findall(text)
        if len(tokens) >= 2:
            t0 = tokens[0].replace(".", ":")
            t1 = tokens[1].replace(".", ":")
            try:
                datetime.strptime(t0, "%H:%M")
                datetime.strptime(t1, "%H:%M")
                return t0, t1
            except:
                pass
        
        return None

    def find_time_by_combination(self, lines: List[Dict], idx: int, max_window: int = 3, 
                                band_top: int = -10**9, band_bottom: int = 10**9) -> Optional[Tuple[int, str, str]]:
        """Поиск времени путем комбинирования соседних строк"""
        n = len(lines)
        
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
        
        if idx - 1 >= 0 and lines[idx - 1]["y"] >= band_top and lines[idx - 1]["y"] <= band_bottom:
            combined = lines[idx - 1]["text"] + " " + lines[idx]["text"]
            res = self.parse_time_in_text(combined)
            if res:
                return (idx - 1, res[0], res[1])
                
        return None

    def find_status_nearby(self, lines: List[Dict], idx: int, band_top: int, band_bottom: int) -> Optional[str]:
        """Поиск статуса рядом с временем"""
        collected = []
        for j in range(idx, min(idx + 6, len(lines))):
            ln = lines[j]
            if ln["y"] < band_top or ln["y"] > band_bottom:
                break
            if j != idx and self.parse_time_in_text(ln["text"]):
                break
            collected.append(ln["text"].lower())
        
        blob = " ".join(collected)
        for key in self.status_map:
            if key in blob:
                return self.status_map[key]
        return None

    def parse_screenshot(self, lines: List[Dict], is_last_screenshot: bool = False) -> List[Dict]:
        """Парсинг одного скриншота"""
        if not lines:
            return []
        
        date_blocks = []
        for idx, ln in enumerate(lines):
            di = self.parse_date_heading(ln["text"])
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
                    
                used_idx, start, end = found
                
                status = self.find_status_nearby(lines, used_idx, band_top=first_y, band_bottom=second_y)
                
                if status == "Отменён":
                    self.cancelled_count += 1
                    continue
                
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
            bands = []
            for i, db in enumerate(date_blocks):
                top_y = db["y"]
                bottom_y = date_blocks[i + 1]["y"] if i + 1 < len(date_blocks) else 10**9
                bands.append((db, top_y, bottom_y))
            
            for i, ln in enumerate(lines):
                if self.date_heading_re.match(ln["text"].strip()):
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
                    
                used_idx, start, end = found
                
                status = self.find_status_nearby(lines, used_idx, band_top=top_y, band_bottom=bottom_y)
                
                if status == "Отменён":
                    self.cancelled_count += 1
                    continue
                
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
        """Обработка всех скриншотов в папке"""
        all_slots = []
        self.cancelled_count = 0
        
        if not os.path.exists(self.base_path):
            logger.warning(f"Path does not exist: {self.base_path}")
            return []
        
        image_files = []
        for file in os.listdir(self.base_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(self.base_path, file))
        
        if not image_files:
            logger.warning(f"No image files found in: {self.base_path}")
            return []
        
        image_files.sort(key=lambda x: os.path.getctime(x))
        
        for idx, fp in enumerate(image_files):
            try:
                lines = self.extract_lines_with_coords(fp)
                
                if not lines:
                    logger.warning(f"No text extracted from {fp}")
                    continue
                    
                is_last = (idx == len(image_files) - 1) and len(image_files) > 1
                slots = self.parse_screenshot(lines, is_last_screenshot=is_last)
                all_slots.extend(slots)
                
            except Exception as e:
                logger.error(f"Error processing screenshot {fp}: {e}")
                # Продолжаем обработку остальных скриншотов
                continue
        
        unique_slots = []
        seen = set()
        for slot in all_slots:
            key = (slot["date"], slot["start"], slot["end"])
            if key not in seen:
                seen.add(key)
                unique_slots.append(slot)
        
        unique_slots.sort(key=lambda s: (s["date"], s["start"]))
        
        # Преобразуем в формат API
        api_slots = []
        for slot in unique_slots:
            api_slots.append({
                "date": slot["date"],
                "startTime": slot["start"],
                "endTime": slot["end"],
                "assignToSelf": True
            })
        
        return api_slots


class MemorySlotParser(SlotParser):
    """Парсер для работы со скриншотами из памяти."""
    
    def __init__(self):
        # Не нужен base_path для работы с памятью
        super().__init__(base_path="")
        self.cancelled_count = 0
        
    def process_screenshot_from_memory(self, image_bytes: BytesIO, is_last: bool = False) -> List[Dict]:
        """Обработка одного скриншота из памяти."""
        try:
            # Читаем изображение из BytesIO
            pil_image = Image.open(image_bytes)
            image_array = np.array(pil_image)
            
            # Если изображение в RGB, конвертируем в BGR для OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Предобработка
            processed = self.preprocess_image_array(image_array)
            if processed is None:
                # Пробуем работать без предобработки
                if len(image_array.shape) == 3:
                    processed = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                else:
                    processed = image_array
            
            # OCR
            try:
                pil_processed = Image.fromarray(processed)
                data = pytesseract.image_to_data(pil_processed, lang="rus", output_type=pytesseract.Output.DICT)
            except Exception as e:
                logger.error(f"OCR error: {e}")
                return []
            
            # Извлекаем строки
            lines = self._extract_lines_from_data(data)
            
            if not lines:
                logger.warning("No lines extracted from memory screenshot")
                return []
            
            # Парсим слоты
            slots = self.parse_screenshot(lines, is_last_screenshot=is_last)
            
            # Преобразуем в формат API
            api_slots = []
            for slot in slots:
                api_slots.append({
                    "date": slot["date"],
                    "startTime": slot["start"],
                    "endTime": slot["end"],
                    "assignToSelf": True
                })
            
            return api_slots
            
        except Exception as e:
            logger.error(f"Error processing screenshot from memory: {e}")
            return []
    
    def preprocess_image_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Предобработка изображения из numpy array."""
        try:
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_array
            
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
            logger.error(f"Error in preprocess_image_array: {e}")
            return None
    
    def _extract_lines_from_data(self, data: Dict) -> List[Dict]:
        """Извлечение строк из OCR данных."""
        try:
            groups = {}
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                groups.setdefault(key, []).append(i)
            
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
            logger.error(f"Error extracting lines from data: {e}")
            return []


# Функция для тестирования модуля отдельно
def main():
    """Основная функция для тестирования парсера"""
    print("🚀 Запуск SlotParser")
    
    # Определяем путь в зависимости от ОС
    if platform.system() == "Windows":
        test_path = r"C:\lavka\lavka\bot\slot"
    else:
        test_path = "./test_screenshots"
    
    print(f"📂 Путь к скриншотам: {test_path}")
    print("ℹ️  Слоты со статусом 'Отменен' будут игнорироваться")
    
    parser = SlotParser(base_path=test_path)
    
    try:
        # Обрабатываем все скриншоты
        slots = parser.process_all_screenshots()
        
        if not slots:
            print("\n❌ Активные слоты не найдены")
            if parser.cancelled_count > 0:
                print(f"   (Все {parser.cancelled_count} найденных слотов были отменены)")
            return
        
        print(f"\n✅ Готово к загрузке: {len(slots)} активных слотов")
        
        # Выводим результат в формате JSON
        print("\n📋 Результат в формате JSON:")
        print("=" * 50)
        print(json.dumps(slots, ensure_ascii=False, indent=2))
        print("=" * 50)
        
        # Показываем примеры слотов
        if slots:
            print("\n📅 Примеры найденных активных слотов:")
            for i, slot in enumerate(slots[:5]):  # Показываем первые 5
                print(f"   {i+1}. {slot['date']} с {slot['startTime']} до {slot['endTime']}")
            if len(slots) > 5:
                print(f"   ... и еще {len(slots) - 5} слотов")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Проверяем наличие необходимых библиотек
    try:
        import cv2
        import numpy
        import PIL
        import pytesseract
    except ImportError as e:
        print("❌ Отсутствуют необходимые библиотеки!")
        print("Установите их командой:")
        print("pip install opencv-python pillow pytesseract numpy")
        exit(1)
    
    # Проверяем путь к Tesseract в зависимости от ОС
    if platform.system() == "Windows":
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            print("❌ Tesseract не найден по указанному пути!")
            print(f"Проверьте путь: {pytesseract.pytesseract.tesseract_cmd}")
            print("Или скачайте Tesseract с: https://github.com/UB-Mannheim/tesseract/wiki")
            exit(1)
    else:
        # На Linux проверяем через which
        import subprocess
        try:
            subprocess.run(["which", "tesseract"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("❌ Tesseract не установлен!")
            print("Установите его командой:")
            print("sudo apt-get install tesseract-ocr tesseract-ocr-rus")
            exit(1)
    
    # Запускаем основную функцию
    main()