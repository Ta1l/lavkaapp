import re
import locale
from datetime import datetime
from typing import List, Dict, Optional, Any

import pytesseract
from PIL import Image

# Для корректного распознавания русских месяцев, устанавливаем локаль.
# Это лучший способ, чем ручной словарь, т.к. учитывает падежи ("августа").
try:
    locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')
except locale.Error:
    print("Внимание: русская локаль ru_RU.UTF-8 не найдена. Парсинг дат может быть неточным.")
    print("Для Windows попробуйте: locale.setlocale(locale.LC_TIME, 'Russian_Russia.1251')")


class SlotParser:
    """
    Класс для парсинга слотов из текста, распознанного с изображения.
    """
    # Более точный паттерн для даты: число, затем слово (месяц)
    # \b - граница слова, чтобы не находить часть слова
    DATE_PATTERN = re.compile(r"^\b(\d{1,2})\s+([а-я]+)\b", re.IGNORECASE)
    
    # Паттерн для времени, статуса и адреса.
    # Захватывает время, затем опционально статус в скобках, и остаток строки как адрес.
    SLOT_PATTERN = re.compile(
        r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})"  # Группы 1 и 2: время начала и конца
        r"(?:\s+\(([^)]+)\))?"                      # Группа 3 (опционально): статус в скобках
        r"\s*(.*)"                                  # Группа 4: остаток строки (адрес)
    )

    def __init__(self, text: str, year: int):
        self.text_lines = text.splitlines()
        self.year = year
        self.current_date: Optional[datetime.date] = None
        self.parsed_slots: List[Dict[str, Any]] = []

    def parse(self) -> List[Dict[str, Any]]:
        """Главный метод, который итерируется по строкам и парсит их."""
        for line in self.text_lines:
            line = line.strip()
            if not line:
                continue

            # Сначала пытаемся распознать строку как дату.
            # Если успешно, обновляем текущую дату и переходим к следующей строке.
            if self._parse_date_from_line(line):
                continue

            # Если это не дата, пытаемся распознать как слот.
            # Для этого должна быть установлена текущая дата.
            if self.current_date:
                self._parse_slot_from_line(line)

        return self.parsed_slots

    def _parse_date_from_line(self, line: str) -> bool:
        """Пытается извлечь дату из строки. Возвращает True в случае успеха."""
        match = self.DATE_PATTERN.match(line)
        if match:
            day_str, month_str = match.groups()
            try:
                # Используем установленную русскую локаль для парсинга названия месяца
                date_str = f"{day_str} {month_str} {self.year}"
                self.current_date = datetime.strptime(date_str, "%d %B %Y").date()
                return True
            except ValueError:
                # Если не удалось распознать месяц, сбрасываем дату
                self.current_date = None
                return False
        return False

    def _parse_slot_from_line(self, line: str):
        """Пытается извлечь детали слота (время, статус, адрес) из строки."""
        match = self.SLOT_PATTERN.search(line)
        if match and self.current_date:
            start_time_str, end_time_str, status, address = match.groups()

            # Если статус не был найден в скобках, пытаемся найти его по ключевым словам
            if not status:
                if "выполнен с опозданием" in line.lower():
                    status = "выполнен с опозданием"
                elif "выполнен" in line.lower():
                    status = "выполнен"
                elif "отмен" in line.lower(): # для "отменен", "отменён"
                    status = "отменён"
                else:
                    status = "неизвестно"
            
            self.parsed_slots.append({
                "date": self.current_date.strftime("%Y-%m-%d"),
                "start_time": start_time_str,
                "end_time": end_time_str,
                "status": status.strip(),
                "address": address.strip() if address else "Адрес не распознан"
            })

def parse_slots_from_image(image_path: str, lang: str = 'rus') -> List[Dict[str, Any]]:
    """
    Извлекает информацию о слотах из изображения.
    
    :param image_path: Путь к файлу изображения.
    :param lang: Язык для распознавания текста (по умолчанию 'rus').
    :return: Список словарей с информацией о слотах.
    """
    try:
        img = Image.open(image_path)
        # Получаем текущий год, чтобы не хардкодить его
        current_year = datetime.now().year
        
        # Распознаем текст с изображения
        text = pytesseract.image_to_string(img, lang=lang)
        
        # Создаем экземпляр парсера и запускаем процесс
        parser = SlotParser(text, year=current_year)
        slots = parser.parse()
        
        return slots

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {image_path}")
        return []
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return []

# === Пример использования ===
if __name__ == '__main__':
    # Указываем путь к твоему новому тестовому файлу
    image_file = 'photo_2025-08-27_01-08-00.jpg'
    
    print(f"Запускаем распознавание для файла: {image_file}\n")
    
    extracted_slots = parse_slots_from_image(image_file)
    
    if extracted_slots:
        print("Распознанные слоты:")
        for i, slot in enumerate(extracted_slots, 1):
            print(f"--- Слот #{i} ---")
            print(f"  Дата: {slot['date']}")
            print(f"  Время: {slot['start_time']} - {slot['end_time']}")
            print(f"  Статус: {slot['status']}")
            print(f"  Адрес: {slot['address']}")
            print("-" * 15)
    else:
        print("Не удалось распознать слоты.")