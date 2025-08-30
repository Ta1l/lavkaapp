import os
import re
import locale
from datetime import datetime
from typing import List, Dict, Optional, Any

import pytesseract
from PIL import Image

# Устанавливаем русскую локаль для корректного парсинга месяцев
try:
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
except locale.Error:
    print("Внимание: Русская локаль ru_RU.UTF-8 не установлена.")

class SlotParser:
    """
    Класс для парсинга слотов из текста, который был распознан с одного или нескольких изображений.
    Адаптирован под новый формат скриншотов.
    """
    # Паттерн для даты остался прежним: "25 августа"
    DATE_PATTERN = re.compile(r"^\b(\d{1,2})\s+([а-я]+)\b", re.IGNORECASE)
    
    # --- ИЗМЕНЕНИЕ: Новый, более простой паттерн для строки слота ---
    # Ищет "ВРЕМЯ - ВРЕМЯ • СТАТУС"
    SLOT_PATTERN = re.compile(
        r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*•\s*(.*)", re.IGNORECASE
    )

    def __init__(self, year: int):
        self.year = year
        self.current_date: Optional[datetime.date] = None
        self.parsed_slots: List[Dict[str, Any]] = []

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """
        Главный метод, который парсит переданный блок текста.
        Может вызываться многократно для текста с разных скриншотов.
        """
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # Если строка - это дата, обновляем текущую дату и идем дальше
            if self._parse_date_from_line(line):
                continue

            # Если это не дата, и у нас есть текущая дата, пытаемся распознать слот
            if self.current_date:
                self._parse_slot_from_line(line)
        
        return self.parsed_slots

    def _parse_date_from_line(self, line: str) -> bool:
        """Пытается извлечь дату из строки. Возвращает True в случае успеха."""
        match = self.DATE_PATTERN.match(line)
        if match:
            day_str, month_str = match.groups()
            try:
                date_str = f"{day_str} {month_str} {self.year}"
                self.current_date = datetime.strptime(date_str, "%d %B %Y").date()
                return True
            except ValueError:
                self.current_date = None
                return False
        return False

    def _parse_slot_from_line(self, line: str):
        """Пытается извлечь детали слота (время и статус) из строки."""
        match = self.SLOT_PATTERN.match(line)
        if match and self.current_date:
            start_time_str, end_time_str, status = match.groups()
            
            self.parsed_slots.append({
                "date": self.current_date.strftime("%Y-%m-%d"),
                "start_time": start_time_str.strip(),
                "end_time": end_time_str.strip(),
                "status": status.strip(),
            })

def process_slot_screenshots(directory: str, lang: str = 'rus') -> List[Dict[str, Any]]:
    """
    Обрабатывает все изображения из указанной папки и извлекает из них слоты.
    
    :param directory: Путь к папке со скриншотами.
    :param lang: Язык для распознавания.
    :return: Единый список всех распознанных слотов.
    """
    if not os.path.isdir(directory):
        print(f"Ошибка: Директория '{directory}' не найдена.")
        return []

    # Получаем текущий год один раз
    current_year = datetime.now().year
    parser = SlotParser(year=current_year)
    
    # Получаем список файлов и сортируем их по имени для правильного порядка
    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Найдено {len(image_files)} изображений для обработки.")

    for filename in image_files:
        image_path = os.path.join(directory, filename)
        try:
            print(f"Обрабатываем файл: {filename}...")
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=lang)
            # Передаем новый текст в тот же экземпляр парсера
            parser.parse(text)
        except Exception as e:
            print(f"Не удалось обработать файл {filename}. Ошибка: {e}")
            
    return parser.parsed_slots

# === Пример использования ===
if __name__ == '__main__':
    # Указываем путь к папке со скриншотами
    screenshots_directory = 'slots'
    
    print(f"Запускаем распознавание для всех файлов в папке: '{screenshots_directory}'\n")
    
    all_extracted_slots = process_slot_screenshots(screenshots_directory)
    
    if all_extracted_slots:
        print("\n--- ИТОГОВЫЙ РЕЗУЛЬТАТ ---")
        print(f"Всего распознано слотов: {len(all_extracted_slots)}")
        for i, slot in enumerate(all_extracted_slots, 1):
            print(f"  {i}. {slot['date']} | {slot['start_time']}-{slot['end_time']} | Статус: {slot['status']}")
    else:
        print("\nНе удалось распознать ни одного слота.")