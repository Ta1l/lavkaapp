# test_ocr_module.py
# -*- coding: utf-8 -*-
"""
Тест-скрипт для проверки работы OCR модуля на 4 скриншотах
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Добавляем путь к модулю если нужно
sys.path.insert(0, r'C:\lavka\lavka')

# ВАЖНО: Устанавливаем путь к Tesseract ДО импорта модуля
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Теперь импортируем OCR модуль
from ocr_module import NewFormatSlotParser, SlotParser

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Константы
TEST_IMAGES_PATH = r"C:\lavka\lavka\test_images"
TEST_FILES = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]

class OCRModuleTester:
    """Класс для тестирования OCR модуля"""
    
    def __init__(self, test_path: str, debug: bool = False):
        self.test_path = test_path
        self.debug = debug
        self.parser = NewFormatSlotParser(debug=debug)
        self.results = {}
        self.statistics = {
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "total_slots": 0,
            "images_with_day": 0,
            "images_with_slots": 0
        }
    
    def test_single_image(self, image_path: str) -> Dict[str, Any]:
        """Тестирование одного изображения"""
        
        filename = os.path.basename(image_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"Тестирование: {filename}")
        logger.info(f"{'='*60}")
        
        result = {
            "filename": filename,
            "path": image_path,
            "success": False,
            "day_extracted": None,
            "slots_found": [],
            "error": None,
            "processing_time": None
        }
        
        if not os.path.exists(image_path):
            result["error"] = f"Файл не найден: {image_path}"
            logger.error(result["error"])
            return result
        
        try:
            start_time = datetime.now()
            
            # Обрабатываем изображение
            slots = self.parser.process_image(image_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time"] = processing_time
            
            if slots:
                result["success"] = True
                result["slots_found"] = slots
                
                # Извлекаем день из первого слота
                if slots and len(slots) > 0:
                    date_str = slots[0].get("date", "")
                    if date_str:
                        try:
                            # Парсим дату вида "2025-10-14"
                            day = int(date_str.split("-")[2])
                            result["day_extracted"] = day
                        except:
                            pass
                
                logger.info(f"✅ Успешно обработано")
                logger.info(f"📅 День: {result['day_extracted']}")
                logger.info(f"⏰ Найдено слотов: {len(slots)}")
                logger.info(f"⏱️ Время обработки: {processing_time:.2f} сек")
                
                # Выводим слоты
                for i, slot in enumerate(slots, 1):
                    logger.info(f"  Слот {i}: {slot['startTime']} - {slot['endTime']} ({slot['date']})")
            else:
                result["error"] = "Слоты не найдены"
                logger.warning("⚠️ Слоты не найдены")
                logger.info(f"⏱️ Время обработки: {processing_time:.2f} сек")
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Ошибка: {e}")
            
        return result
    
    def test_all_images(self) -> Dict[str, Any]:
        """Тестирование всех изображений"""
        
        logger.info("\n" + "="*60)
        logger.info("НАЧАЛО ТЕСТИРОВАНИЯ OCR МОДУЛЯ")
        logger.info("="*60)
        
        all_results = []
        
        for filename in TEST_FILES:
            image_path = os.path.join(self.test_path, filename)
            result = self.test_single_image(image_path)
            all_results.append(result)
            
            # Обновляем статистику
            self.statistics["total_images"] += 1
            
            if result["success"]:
                self.statistics["successful"] += 1
                
                if result["day_extracted"]:
                    self.statistics["images_with_day"] += 1
                
                if result["slots_found"]:
                    self.statistics["images_with_slots"] += 1
                    self.statistics["total_slots"] += len(result["slots_found"])
            else:
                self.statistics["failed"] += 1
        
        return {
            "test_time": datetime.now().isoformat(),
            "results": all_results,
            "statistics": self.statistics
        }
    
    def print_summary(self, test_results: Dict[str, Any]):
        """Вывод итоговой статистики"""
        
        stats = test_results["statistics"]
        
        logger.info("\n" + "="*60)
        logger.info("ИТОГОВАЯ СТАТИСТИКА")
        logger.info("="*60)
        
        logger.info(f"📊 Всего изображений: {stats['total_images']}")
        logger.info(f"✅ Успешно обработано: {stats['successful']}")
        logger.info(f"❌ Ошибок: {stats['failed']}")
        logger.info(f"📅 Изображений с определенным днем: {stats['images_with_day']}")
        logger.info(f"⏰ Изображений со слотами: {stats['images_with_slots']}")
        logger.info(f"📋 Всего найдено слотов: {stats['total_slots']}")
        
        success_rate = (stats['successful'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
        logger.info(f"📈 Процент успеха: {success_rate:.1f}%")
        
        # Выводим сводку по дням
        logger.info("\n" + "="*60)
        logger.info("СВОДКА ПО ДНЯМ")
        logger.info("="*60)
        
        for result in test_results["results"]:
            if result["day_extracted"]:
                slots_count = len(result["slots_found"])
                logger.info(f"📅 {result['filename']}: День {result['day_extracted']}, слотов: {slots_count}")
                if result["slots_found"]:
                    for slot in result["slots_found"]:
                        logger.info(f"   • {slot['startTime']} - {slot['endTime']}")
            else:
                logger.info(f"❌ {result['filename']}: День не определен")
    
    def export_json(self, test_results: Dict[str, Any], output_file: str = "test_results.json"):
        """Экспорт результатов в JSON"""
        
        # Подготавливаем данные для JSON (убираем лишнее)
        json_data = {
            "test_date": test_results["test_time"],
            "statistics": test_results["statistics"],
            "images": []
        }
        
        for result in test_results["results"]:
            image_data = {
                "filename": result["filename"],
                "day": result["day_extracted"],
                "slots": []
            }
            
            if result["slots_found"]:
                for slot in result["slots_found"]:
                    image_data["slots"].append({
                        "date": slot["date"],
                        "startTime": slot["startTime"],
                        "endTime": slot["endTime"]
                    })
            
            json_data["images"].append(image_data)
        
        # Сохраняем в файл
        output_path = os.path.join(self.test_path, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n💾 Результаты сохранены в: {output_path}")
        
        # Также выводим JSON в консоль
        logger.info("\n" + "="*60)
        logger.info("JSON РЕЗУЛЬТАТЫ")
        logger.info("="*60)
        print(json.dumps(json_data, ensure_ascii=False, indent=2))
        
        return json_data

def test_batch_processing():
    """Тест пакетной обработки через SlotParser"""
    
    logger.info("\n" + "="*60)
    logger.info("ТЕСТ ПАКЕТНОЙ ОБРАБОТКИ (SlotParser)")
    logger.info("="*60)
    
    parser = SlotParser(TEST_IMAGES_PATH, debug=False)
    slots = parser.process_all_screenshots()
    
    if slots:
        logger.info(f"✅ Найдено {len(slots)} уникальных слотов")
        
        # Группируем по датам
        slots_by_date = {}
        for slot in slots:
            date = slot["date"]
            if date not in slots_by_date:
                slots_by_date[date] = []
            slots_by_date[date].append(slot)
        
        logger.info(f"📅 Найдено {len(slots_by_date)} уникальных дней")
        
        # Выводим по датам
        for date in sorted(slots_by_date.keys()):
            day_slots = slots_by_date[date]
            day = int(date.split("-")[2])
            logger.info(f"\nДень {day} ({date}):")
            for slot in day_slots:
                logger.info(f"  • {slot['startTime']} - {slot['endTime']}")
        
        return slots
    else:
        logger.warning("⚠️ Слоты не найдены")
        return []

def check_tesseract():
    """Проверка доступности Tesseract"""
    try:
        import pytesseract
        
        # Устанавливаем путь к Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Проверяем что он работает
        version = pytesseract.get_tesseract_version()
        logger.info(f"✅ Tesseract доступен")
        logger.info(f"   Путь: {pytesseract.pytesseract.tesseract_cmd}")
        logger.info(f"   Версия: {version}")
        return True
    except Exception as e:
        logger.error(f"❌ Tesseract недоступен: {e}")
        logger.error("Установите Tesseract OCR для Windows:")
        logger.error("1. Скачайте с: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.error("2. Установите в C:\\Program Files\\Tesseract-OCR\\")
        return False

def main():
    """Главная функция тестирования"""
    
    print("\n" + "="*60)
    print("OCR MODULE TESTER v1.0")
    print("="*60)
    
    # Проверяем Tesseract
    if not check_tesseract():
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Tesseract не работает!")
        print("Проверьте установку и путь.")
        return
    
    # Проверяем наличие тестовых изображений
    if not os.path.exists(TEST_IMAGES_PATH):
        logger.error(f"❌ Папка с тестовыми изображениями не найдена: {TEST_IMAGES_PATH}")
        return
    
    # Проверяем наличие файлов
    missing_files = []
    for filename in TEST_FILES:
        path = os.path.join(TEST_IMAGES_PATH, filename)
        if not os.path.exists(path):
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"❌ Не найдены файлы: {', '.join(missing_files)}")
        return
    
    logger.info(f"📁 Папка с тестами: {TEST_IMAGES_PATH}")
    logger.info(f"📷 Файлы для тестирования: {', '.join(TEST_FILES)}")
    
    # Создаем тестер
    tester = OCRModuleTester(TEST_IMAGES_PATH, debug=False)
    
    # Запускаем тесты
    test_results = tester.test_all_images()
    
    # Выводим статистику
    tester.print_summary(test_results)
    
    # Экспортируем в JSON
    json_data = tester.export_json(test_results)
    
    # Дополнительно тестируем пакетную обработку
    print("\n")
    batch_slots = test_batch_processing()
    
    # Сравниваем результаты
    logger.info("\n" + "="*60)
    logger.info("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    logger.info("="*60)
    
    individual_slots = sum(len(r["slots_found"]) for r in test_results["results"])
    batch_slots_count = len(batch_slots)
    
    logger.info(f"Слотов при индивидуальной обработке: {individual_slots}")
    logger.info(f"Слотов при пакетной обработке: {batch_slots_count}")
    
    if individual_slots == batch_slots_count:
        logger.info("✅ Результаты совпадают")
    else:
        logger.warning("⚠️ Результаты различаются")
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)
    
    # Финальная проверка
    if test_results["statistics"]["successful"] == 0:
        print("\n⚠️ ВНИМАНИЕ: Ни одно изображение не было обработано успешно!")
        print("Возможные причины:")
        print("1. Проблемы с форматом изображений")
        print("2. Изображения не содержат ожидаемых элементов")
        print("3. Требуется настройка параметров OCR")

if __name__ == "__main__":
    main()