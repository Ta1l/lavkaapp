# server_test.py
import sys
import os
from io import BytesIO

# Проверка Tesseract
try:
    import pytesseract
    print(f"✅ Pytesseract imported")
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
except Exception as e:
    print(f"❌ Pytesseract error: {e}")
    sys.exit(1)

# Проверка OpenCV
try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV error: {e}")
    sys.exit(1)

# Тест OCR модуля
try:
    from ocr_module import MemorySlotParser
    print("✅ OCR module imported")
    
    # Тест с изображением если есть
    test_image = "test.jpg"  # Положите тестовое изображение
    if os.path.exists(test_image):
        with open(test_image, 'rb') as f:
            image_bytes = BytesIO(f.read())
        
        parser = MemorySlotParser(debug=True)
        slots = parser.process_image(image_bytes)
        print(f"Found {len(slots)} slots: {slots}")
    else:
        print("No test image found")
        
except Exception as e:
    print(f"❌ OCR module error: {e}")
    import traceback
    traceback.print_exc()