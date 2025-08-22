import pytesseract
from PIL import Image
import re
import json

# путь к tesseract (обычно он и так в PATH, но если нет - раскомментируй)
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

def parse_slots(image_path: str):
    # Загружаем картинку
    img = Image.open(image_path)

    # Распознаём текст
    text = pytesseract.image_to_string(img, lang="rus")

    slots = []
    current_day = None

    # Разбиваем на строки
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # День недели + дата
        if re.match(r"\d{1,2}\s+\w+", line.lower()):
            current_day = line
            continue

        # Время + статус
        time_match = re.search(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", line)
        if time_match:
            slot = {
                "day": current_day,
                "time": f"{time_match.group(1)} - {time_match.group(2)}",
                "status": None
            }

            # определяем статус
            if "выполнен" in line.lower():
                slot["status"] = "выполнен"
            elif "отмен" in line.lower():
                slot["status"] = "отменён"
            elif "опоздан" in line.lower():
                slot["status"] = "выполнен с опозданием"
            else:
                slot["status"] = "неизвестно"

            slots.append(slot)

    return slots


if __name__ == "__main__":
    result = parse_slots("4d215fd6-3b56-44d3-abde-603a906ab546.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))
