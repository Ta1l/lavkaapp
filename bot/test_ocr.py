# test_ocr_debug_fixed.py
# -*- coding: utf-8 -*-
"""
Исправленная диагностика OCR с корректной обработкой регионов
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import date
from typing import List, Dict, Optional, Tuple

# Добавляем путь к модулю
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_module import NewFormatSlotParser
import cv2
import pytesseract
from PIL import Image
import numpy as np

# Настройка Tesseract для Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Константы для даты
FIXED_YEAR = 2025
FIXED_MONTH = 10


def find_time_slots(text: str) -> List[Tuple[str, str]]:
    """Находит временные слоты в тексте"""
    slots = []
    
    # Очищаем текст от лишних символов
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Расширенные паттерны для поиска времени
    time_patterns = [
        # Стандартные форматы с двоеточием
        r'(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})',
        # С точкой
        r'(\d{1,2})\.(\d{2})\s*[-–—]\s*(\d{1,2})\.(\d{2})',
        # С пробелом
        r'(\d{1,2})\s+(\d{2})\s*[-–—]\s*(\d{1,2})\s+(\d{2})',
        # Слитно (0800-1300)
        r'(\d{2})(\d{2})\s*[-–—]\s*(\d{2})(\d{2})',
    ]
    
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if len(match) == 4:
                    h1, m1, h2, m2 = match
                    h1, m1, h2, m2 = int(h1), int(m1), int(h2), int(m2)
                    
                    if 0 <= h1 < 24 and 0 <= m1 < 60 and 0 <= h2 < 24 and 0 <= m2 < 60:
                        start_time = f"{h1:02d}:{m1:02d}"
                        end_time = f"{h2:02d}:{m2:02d}"
                        if (start_time, end_time) not in slots:
                            slots.append((start_time, end_time))
            except:
                continue
    
    return slots


def extract_day_simple(img: np.ndarray) -> Optional[int]:
    """Простое извлечение дня из изображения"""
    try:
        # Берем верхнюю часть изображения
        h = img.shape[0]
        top_part = img[:h//3, :]
        
        # Конвертируем в PIL
        pil_img = Image.fromarray(cv2.cvtColor(top_part, cv2.COLOR_BGR2RGB))
        
        # OCR
        text = pytesseract.image_to_string(pil_img, lang='eng+rus')
        
        # Ищем числа от 1 до 31
        numbers = re.findall(r'\b(\d{1,2})\b', text)
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 31:
                return num
    except Exception as e:
        print(f"Ошибка извлечения дня: {e}")
    
    return None


def analyze_image_for_slots(image_path: str) -> List[Dict]:
    """Анализирует изображение и возвращает слоты в JSON формате"""
    print(f"\n{'='*60}")
    print(f"🔍 АНАЛИЗ СЛОТОВ: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Читаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Не удалось прочитать изображение")
        return []
    
    print(f"📐 Размер изображения: {img.shape[1]}x{img.shape[0]}")
    
    # 1. Извлекаем день (упрощенный метод)
    day = extract_day_simple(img)
    if day is None:
        # Пробуем через основной модуль
        try:
            parser = NewFormatSlotParser(debug=False)
            day = parser._ocr_day_fallback_topstrip(img)
        except:
            pass
    
    if day is None:
        print("⚠️ День не определен, используем день по умолчанию: 1")
        day = 1
    else:
        print(f"✅ Определен день: {day}")
    
    # Формируем дату
    try:
        slot_date = date(FIXED_YEAR, FIXED_MONTH, day)
        iso_date = slot_date.isoformat()
        print(f"📅 Дата: {iso_date}")
    except:
        print("❌ Ошибка формирования даты")
        return []
    
    # 2. Ищем временные слоты
    all_slots = []
    seen_slots = set()
    
    # Способ 1: Простое распознавание всего изображения
    print("\n📝 Распознавание текста...")
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    text_simple = pytesseract.image_to_string(pil_img, lang='eng+rus')
    
    print("Найденный текст (первые 500 символов):")
    print(text_simple[:500])
    
    # Ищем слоты
    slots = find_time_slots(text_simple)
    print(f"\nНайдено временных диапазонов: {len(slots)}")
    
    for start, end in slots:
        key = (start, end)
        if key not in seen_slots:
            seen_slots.add(key)
            all_slots.append({
                "date": iso_date,
                "startTime": start,
                "endTime": end,
                "assignToSelf": True
            })
            print(f"  • {start} - {end}")
    
    # Способ 2: С предобработкой
    print("\n📝 Распознавание с предобработкой...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Бинаризация
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    binary_pil = Image.fromarray(binary)
    text_processed = pytesseract.image_to_string(binary_pil, lang='eng+rus')
    
    slots = find_time_slots(text_processed)
    print(f"Найдено с предобработкой: {len(slots)}")
    
    for start, end in slots:
        key = (start, end)
        if key not in seen_slots:
            seen_slots.add(key)
            all_slots.append({
                "date": iso_date,
                "startTime": start,
                "endTime": end,
                "assignToSelf": True
            })
            print(f"  • {start} - {end}")
    
    # Способ 3: По областям (безопасный)
    print("\n📝 Анализ по областям...")
    h, w = img.shape[:2]
    
    # Определяем безопасные регионы
    regions = [
        ("Центр", 0, h//3, w, 2*h//3),
        ("Низ", 0, 2*h//3, w, h),
    ]
    
    for name, x1, y1, x2, y2 in regions:
        # Проверяем границы
        x1 = max(0, min(x1, w-1))
        x2 = max(x1+1, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(y1+1, min(y2, h))
        
        if x2 > x1 and y2 > y1:
            roi = img[y1:y2, x1:x2]
            
            if roi.size > 0:
                try:
                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    text_roi = pytesseract.image_to_string(roi_pil, lang='eng+rus')
                    
                    slots = find_time_slots(text_roi)
                    if slots:
                        print(f"  В области '{name}' найдено: {len(slots)}")
                        
                    for start, end in slots:
                        key = (start, end)
                        if key not in seen_slots:
                            seen_slots.add(key)
                            all_slots.append({
                                "date": iso_date,
                                "startTime": start,
                                "endTime": end,
                                "assignToSelf": True
                            })
                            print(f"    • {start} - {end}")
                except Exception as e:
                    print(f"  Ошибка обработки области '{name}': {e}")
    
    # Сортируем по времени начала
    all_slots.sort(key=lambda x: x["startTime"])
    
    print(f"\n📊 Итого найдено уникальных слотов: {len(all_slots)}")
    
    return all_slots


def analyze_text_patterns(image_path: str):
    """Анализ паттернов для отладки"""
    print(f"\n{'='*60}")
    print(f"🔍 ПОИСК ВРЕМЕННЫХ ПАТТЕРНОВ")
    print(f"{'='*60}")
    
    img = cv2.imread(image_path)
    if img is None:
        return
    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(pil_img, lang='eng+rus')
    
    print("\n📝 ЧИСЛОВЫЕ ПАТТЕРНЫ В ТЕКСТЕ:")
    print("-" * 40)
    
    # Различные паттерны
    patterns = {
        "XX:XX": r'\d{1,2}:\d{2}',
        "XX.XX": r'\d{1,2}\.\d{2}',
        "XX XX": r'\d{1,2}\s+\d{2}',
        "XXXX": r'\b\d{4}\b',
        "XX-XX": r'\d{1,2}-\d{2}',
        "XX,XX": r'\d{1,2},\d{2}',
    }
    
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            print(f"{name}: {matches[:10]}")  # первые 10
    
    # Поиск строк содержащих числа и тире
    print("\n📝 СТРОКИ С ЧИСЛАМИ И РАЗДЕЛИТЕЛЯМИ:")
    print("-" * 40)
    
    lines = text.split('\n')
    for line in lines:
        # Если в строке есть числа и какой-то разделитель
        if re.search(r'\d', line) and any(sep in line for sep in ['-', '–', '—', ':', '.', ',']):
            print(f"  {line[:100]}")  # первые 100 символов строки


def main():
    """Главная функция"""
    print("🚀 ДИАГНОСТИКА OCR С JSON ВЫВОДОМ (ИСПРАВЛЕННАЯ)")
    
    # Путь к изображению
    test_image = r"C:\lavka\lavka\test_images\photo_2025-10-11_14-36-35.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Файл не найден: {test_image}")
        print("Укажите путь к вашему изображению:")
        test_image = input("> ").strip()
    
    if os.path.exists(test_image):
        # Анализ паттернов
        analyze_text_patterns(test_image)
        
        # Основной анализ слотов
        slots = analyze_image_for_slots(test_image)
        
        # Вывод результата в JSON
        print(f"\n{'='*60}")
        print("📊 РЕЗУЛЬТАТ В JSON ФОРМАТЕ:")
        print(f"{'='*60}")
        
        if slots:
            print(json.dumps(slots, ensure_ascii=False, indent=2))
        else:
            print("[]")
        
        # Сохраняем результат
        output_dir = Path("ocr_debug_output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "slots.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(slots, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Результат сохранен в: {output_file.absolute()}")
        
        # Сохраняем полный текст для анализа
        img = cv2.imread(test_image)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        full_text = pytesseract.image_to_string(pil_img, lang='eng+rus')
        
        with open(output_dir / "full_text.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        
        print(f"✅ Полный текст сохранен в: {output_dir / 'full_text.txt'}")
        
    else:
        print(f"❌ Файл не найден: {test_image}")


if __name__ == "__main__":
    main()