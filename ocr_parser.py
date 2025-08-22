import pytesseract
from PIL import Image
import re
from datetime import datetime

def parse_slots(image_path: str):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="rus")

    slots = []
    current_date = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Дата (например "18 августа, понедельник")
        if re.match(r"\d{1,2}\s+\w+", line.lower()):
            try:
                # вытаскиваем только дату
                date_str = line.split(",")[0]
                current_date = datetime.strptime(date_str + " 2025", "%d %B %Y").date()
            except Exception:
                current_date = None
            continue

        # Время
        time_match = re.search(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", line)
        if time_match and current_date:
            start_time = datetime.strptime(f"{current_date} {time_match.group(1)}", "%Y-%m-%d %H:%M")
            end_time = datetime.strptime(f"{current_date} {time_match.group(2)}", "%Y-%m-%d %H:%M")

            status = "неизвестно"
            if "выполнен с опозданием" in line.lower():
                status = "выполнен с опозданием"
            elif "выполнен" in line.lower():
                status = "выполнен"
            elif "отмен" in line.lower():
                status = "отменён"

            slots.append({
                "date": str(current_date),
                "start_time": start_time,
                "end_time": end_time,
                "status": status,
                "address": "Северный проспект, д. 4 корп. 1"  # можно дораспознавать
            })

    return slots
