# -*- coding: utf-8 -*-
"""
Slot OCR parser — improved: combine neighboring lines & fallback scan for times.
Requirements:
  pip install pytesseract pillow opencv-python numpy
Configure tesseract path below.
Put screenshots 1.png .. 6.png in C:\lavka\lavka\slots (or change base_path).
"""

import os
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract

# Укажи путь к tesseract.exe (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class SlotParser:
    def __init__(self, base_path: str = r"C:\lavka\lavka\slots"):
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

        # статусы
        self.status_map = {
            "выполнен с опозданием": "Выполнен с опозданием",
            "выполнен": "Выполнен",
            "отменен": "Отменён",
            "отменён": "Отменён",
            "отмен": "Отменён"
        }

        months_alt = "|".join(self.months.keys())
        weekdays_alt = "|".join(self.weekdays.keys())

        # строгий заголовок даты: "25 августа, понедельник"
        self.date_heading_re = re.compile(
            rf"^\s*(\d{{1,2}})\s+({months_alt})\s*,\s*({weekdays_alt})\s*$",
            flags=re.IGNORECASE
        )

        # верхний диапазон недели (игнорируем)
        self.week_range_re = re.compile(
            rf"\d{{1,2}}\s+({months_alt})\s+\d{{4}}\s*[-–—]\s*\d{{1,2}}\s+({months_alt})\s+\d{{4}}",
            flags=re.IGNORECASE
        )

        # базовая "поиск одной метки времени" (HH:MM или HH.MM)
        self.time_token_re = re.compile(r"(\d{1,2}[:.]\d{2})")

        # временная папка
        self.temp_dir = os.path.join(self.base_path, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    # ---------- предобработка изображения ----------
    def preprocess_image(self, image_path: str) -> str:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # немного увеличим
            scale = 1.4
            resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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

    # ---------- OCR -> строки с координатами ----------
    def extract_lines_with_coords(self, image_path: str) -> List[Dict]:
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
            if proc != image_path and os.path.exists(proc):
                try:
                    os.remove(proc)
                except:
                    pass
            return lines
        except Exception as e:
            print(f"[extract_lines_with_coords] OCR error: {e}")
            return []

    # ---------- парсинг дат/времени/статуса ----------
    def _pick_year_from_week_range(self, lines: List[Dict]) -> Optional[int]:
        for ln in lines[:6]:
            m = re.search(r"\b(\d{4})\b", ln["text"])
            if m:
                try:
                    return int(m.group(1))
                except:
                    pass
            if self.week_range_re.search(ln["text"]):
                yrs = re.findall(r"\d{4}", ln["text"])
                if yrs:
                    try:
                        return int(yrs[-1])
                    except:
                        pass
        return None

    def parse_date_heading(self, text: str, fallback_year: Optional[int]) -> Optional[Dict]:
        m = self.date_heading_re.match(text.strip())
        if not m:
            return None
        day = int(m.group(1))
        month_name = m.group(2).lower()
        weekday_name = m.group(3).lower()
        month = self.months.get(month_name)
        year = fallback_year or datetime.now().year
        try:
            date_obj = datetime(year, month, day)
        except:
            return None
        return {"date": date_obj.strftime("%Y-%m-%d"), "weekday": weekday_name, "y": None, "line_idx": None}

    def parse_time_in_text(self, text: str) -> Optional[Tuple[str, str]]:
        """Ищем в тексте все токены времени, если >=2 -- возвращаем первые два (нормализованные)."""
        tokens = self.time_token_re.findall(text)
        if len(tokens) < 2:
            return None
        # нормализуем (заменяем точку на двоеточие)
        t0 = tokens[0].replace(".", ":")
        t1 = tokens[1].replace(".", ":")
        try:
            datetime.strptime(t0, "%H:%M")
            datetime.strptime(t1, "%H:%M")
            return t0, t1
        except:
            return None

    def find_time_by_combination(self, lines: List[Dict], idx: int, max_window: int = 3, band_top: int = -10**9, band_bottom: int = 10**9) -> Optional[Tuple[int, str, str]]:
        """
        Попытка найти пару времени, комбинируя данную строку с последующими (до max_window).
        Возвращает (used_index, start, end) или None.
        Обрезаем комбинирование, если следующая строка вышла за band_top..band_bottom.
        """
        n = len(lines)
        # попробуем комбинировать curr, curr+1, curr+2...
        for w in range(1, max_window + 1):
            j = idx + w - 1
            if j >= n:
                break
            # проверка вертикально — если next line вышла за границу полосы — прерываем
            if lines[j]["y"] < band_top or lines[j]["y"] > band_bottom:
                break
            combined = " ".join(lines[k]["text"] for k in range(idx, j + 1))
            res = self.parse_time_in_text(combined)
            if res:
                return (idx, res[0], res[1])
        # также пробуем prev+curr (некоторые times могут начинаться на предыдущей линейке)
        if idx - 1 >= 0 and lines[idx - 1]["y"] >= band_top and lines[idx - 1]["y"] <= band_bottom:
            combined = lines[idx - 1]["text"] + " " + lines[idx]["text"]
            res = self.parse_time_in_text(combined)
            if res:
                return (idx - 1, res[0], res[1])
        return None

    def find_status_nearby(self, lines: List[Dict], idx: int, band_top: int, band_bottom: int) -> Optional[str]:
        """
        Ищем статус начиная с строки idx вниз в пределах band_top..band_bottom.
        Останавливаем, если встречаем новую временную метку (иначе можем переполучить статус следующего слота).
        """
        collected = []
        for j in range(idx, min(idx + 6, len(lines))):
            ln = lines[j]
            if ln["y"] < band_top or ln["y"] > band_bottom:
                break
            # если видим новую метку времени (и это не наша первая строка) — прерываем
            if j != idx and self.parse_time_in_text(ln["text"]):
                break
            collected.append(ln["text"].lower())
        blob = " ".join(collected)
        for key in self.status_map:
            if key in blob:
                return self.status_map[key]
        return None

    # ---------- основной парсер одного скрина ----------
    def parse_screenshot(self, lines: List[Dict], is_last_screenshot: bool = False) -> List[Dict]:
        if not lines:
            return []
        year_guess = self._pick_year_from_week_range(lines)

        # находим заголовки дат
        date_blocks: List[Dict] = []
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
        slots: List[Dict] = []
        seen_keys = set()

        if not is_last_screenshot:
            # непоследний: работаем только с первой датой
            first = date_blocks[0]
            first_y = first["y"]
            first_idx = first["line_idx"]
            if len(date_blocks) >= 2:
                second_y = date_blocks[1]["y"]
                second_idx = date_blocks[1]["line_idx"]
            else:
                second_y = 10**9
                second_idx = len(lines) + 1

            for i in range(first_idx + 1, min(second_idx, len(lines))):
                ln = lines[i]
                if ln["y"] <= first_y or ln["y"] >= second_y:
                    continue
                # ищем время, комбинируя соседние линии
                found = self.find_time_by_combination(lines, i, max_window=3, band_top=first_y, band_bottom=second_y)
                if not found:
                    continue
                used_idx, start, end = found
                # статус
                status = self.find_status_nearby(lines, used_idx, band_top=first_y, band_bottom=second_y)
                slot = {
                    "date": first["date"],
                    "weekday": first["weekday"],
                    "start_time": start,
                    "end_time": end,
                    "status": status if status is not None else "Неизвестно"
                }
                key = (slot["date"], slot["start_time"], slot["end_time"], slot["status"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # длительность
                try:
                    t1 = datetime.strptime(start, "%H:%M")
                    t2 = datetime.strptime(end, "%H:%M")
                    if t2 < t1:
                        t2 += timedelta(days=1)
                    slot["duration_minutes"] = int((t2 - t1).total_seconds() // 60)
                except:
                    slot["duration_minutes"] = 0
                slots.append(slot)
            return slots

        else:
            # последний скрин: создаём полосы между датами (bands)
            bands = []
            for i, db in enumerate(date_blocks):
                top_y = db["y"]
                bottom_y = date_blocks[i + 1]["y"] if i + 1 < len(date_blocks) else 10**9
                bands.append((db, top_y, bottom_y))

            # первоначальный проход: ищем временные пары, комбинируя строки, в пределах band.
            for i, ln in enumerate(lines):
                if self.week_range_re.search(ln["text"]):
                    continue
                if self.date_heading_re.match(ln["text"].strip()):
                    continue
                # определить к какой полосе принадлежит ln
                y = ln["y"]
                assigned = None
                for db, top_y, bottom_y in bands:
                    if y > top_y and y < bottom_y:
                        assigned = (db, top_y, bottom_y)
                        break
                if not assigned:
                    continue
                db, top_y, bottom_y = assigned
                found = self.find_time_by_combination(lines, i, max_window=3, band_top=top_y, band_bottom=bottom_y)
                if not found:
                    continue
                used_idx, start, end = found
                status = self.find_status_nearby(lines, used_idx, band_top=top_y, band_bottom=bottom_y)
                slot = {
                    "date": db["date"],
                    "weekday": db["weekday"],
                    "start_time": start,
                    "end_time": end,
                    "status": status if status is not None else "Неизвестно"
                }
                key = (slot["date"], slot["start_time"], slot["end_time"], slot["status"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                try:
                    t1 = datetime.strptime(start, "%H:%M")
                    t2 = datetime.strptime(end, "%H:%M")
                    if t2 < t1:
                        t2 += timedelta(days=1)
                    slot["duration_minutes"] = int((t2 - t1).total_seconds() // 60)
                except:
                    slot["duration_minutes"] = 0
                slots.append(slot)

            # fallback: для каждой полосы, если какие-то пары времени ещё не найдены (например OCR сильно разбил),
            # делаем второй скан: склеиваем окна из 1..3 строк и ищем пары.
            for db, top_y, bottom_y in bands:
                # собрать индексы строк, принадлежащих band
                idxs = [idx for idx, ln in enumerate(lines) if ln["y"] > top_y and ln["y"] < bottom_y]
                for idx in idxs:
                    # если в текуще band уже есть пара с такими start/end — skip (dupe prevented by seen_keys)
                    found = self.find_time_by_combination(lines, idx, max_window=3, band_top=top_y, band_bottom=bottom_y)
                    if not found:
                        continue
                    used_idx, start, end = found
                    # уже добавлена?
                    key_check = (db["date"], start, end)
                    # check duplicates by only date/start/end ignoring status to catch same time with different status
                    already = any((s["date"] == db["date"] and s["start_time"] == start and s["end_time"] == end) for s in slots)
                    if already:
                        continue
                    status = self.find_status_nearby(lines, used_idx, band_top=top_y, band_bottom=bottom_y)
                    slot = {
                        "date": db["date"],
                        "weekday": db["weekday"],
                        "start_time": start,
                        "end_time": end,
                        "status": status if status is not None else "Неизвестно"
                    }
                    key = (slot["date"], slot["start_time"], slot["end_time"], slot["status"])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    try:
                        t1 = datetime.strptime(start, "%H:%M")
                        t2 = datetime.strptime(end, "%H:%M")
                        if t2 < t1:
                            t2 += timedelta(days=1)
                        slot["duration_minutes"] = int((t2 - t1).total_seconds() // 60)
                    except:
                        slot["duration_minutes"] = 0
                    slots.append(slot)

            return slots

    # ---------- процесс всех 6 скринов ----------
    def process_all_screenshots(self) -> List[Dict]:
        all_slots: List[Dict] = []
        for n in range(1, 7):
            fp = None
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                candidate = os.path.join(self.base_path, f"{n}{ext}")
                if os.path.exists(candidate):
                    fp = candidate
                    break
            if not fp:
                print(f"⚠️  Файл {n} не найден — пропускаю.")
                continue
            print(f"\n📄 Обработка {os.path.basename(fp)}")
            lines = self.extract_lines_with_coords(fp)
            print(f"   Строк извлечено: {len(lines)}")
            is_last = (n == 6)
            slots = self.parse_screenshot(lines, is_last_screenshot=is_last)
            print(f"   Найдено слотов на скрине: {len(slots)}")
            all_slots.extend(slots)
        all_slots.sort(key=lambda s: (s["date"], s["start_time"]))
        return all_slots

    def save_results(self, slots: List[Dict]) -> Tuple[str, str]:
        grouped = {}
        for s in slots:
            grouped.setdefault(s["date"], []).append(s)
        out_all = os.path.join(self.base_path, "slots.json")
        out_grouped = os.path.join(self.base_path, "slots_grouped.json")
        with open(out_all, "w", encoding="utf-8") as f:
            json.dump(slots, f, ensure_ascii=False, indent=2)
        with open(out_grouped, "w", encoding="utf-8") as f:
            json.dump(grouped, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Сохранено: {out_all} и {out_grouped}")
        return out_all, out_grouped

    def generate_report(self, slots: List[Dict]) -> Dict:
        report = {
            "total_slots": len(slots),
            "total_hours": round(sum(s.get("duration_minutes", 0) for s in slots) / 60, 2),
            "by_status": {},
            "by_weekday": {},
            "by_date": {},
            "date_range": {
                "start": min((s["date"] for s in slots), default=None),
                "end": max((s["date"] for s in slots), default=None),
            },
        }
        for s in slots:
            st = s.get("status", "Неизвестно")
            report["by_status"][st] = report["by_status"].get(st, 0) + 1
            wd = s.get("weekday", "н/д")
            b = report["by_weekday"].setdefault(wd, {"slots": 0, "minutes": 0})
            b["slots"] += 1
            b["minutes"] += s.get("duration_minutes", 0)
            d = report["by_date"].setdefault(s["date"], {"slots": 0, "minutes": 0})
            d["slots"] += 1
            d["minutes"] += s.get("duration_minutes", 0)
        return report

    def cleanup(self):
        try:
            if os.path.isdir(self.temp_dir):
                for fn in os.listdir(self.temp_dir):
                    try:
                        os.remove(os.path.join(self.temp_dir, fn))
                    except:
                        pass
                try:
                    os.rmdir(self.temp_dir)
                except:
                    pass
        except Exception as e:
            print(f"[cleanup] {e}")

    # ---------- debug ----------
    def debug_screenshot(self, screenshot_number: int):
        fp = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(self.base_path, f"{screenshot_number}{ext}")
            if os.path.exists(cand):
                fp = cand
                break
        if not fp:
            print("Файл не найден:", screenshot_number)
            return
        lines = self.extract_lines_with_coords(fp)
        print(f"\n--- lines for {fp} ---")
        for i, ln in enumerate(lines):
            print(f"{i:03d} | y={ln['y']:4d} | x={ln['x']:4d} | {ln['text']}")
        slots = self.parse_screenshot(lines, is_last_screenshot=(screenshot_number == 6))
        print(f"\n--- parsed slots ({len(slots)}) ---")
        for s in slots:
            print(s)


def main():
    print("Запуск SlotParser (improved)")
    parser = SlotParser()
    try:
        slots = parser.process_all_screenshots()
        if not slots:
            print("❌ Слоты не найдены.")
            return
        print(f"\n✅ Всего слотов: {len(slots)}")
        for s in slots:
            print(f" {s['date']} {s['weekday']}: {s['start_time']} - {s['end_time']} ({s['status']}) [{s.get('duration_minutes',0)}m]")
        parser.save_results(slots)
        report = parser.generate_report(slots)
        report_path = os.path.join(parser.base_path, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📊 Отчёт: {report_path}")
    finally:
        parser.cleanup()
        print("Готово.")


if __name__ == "__main__":
    main()