// utils/calcDayHours.ts
// Утилита: получает массив слотов { startTime: "HH:MM", endTime: "HH:MM" }
// Возвращает суммарное время в часах (number, округлено до 2 знаков при необходимости)

export interface SimpleSlot {
    startTime: string; // "08:00"
    endTime: string;   // "15:30"
  }
  
  /**
   * Преобразует "HH:MM" -> минуты с начала дня (0..)
   * Если end < start — считаем, что слот перешёл через полночь (end += 24*60).
   */
  function timeToMinutes(time: string): number {
    const [hhStr, mmStr] = time.split(":");
    const hh = parseInt(hhStr, 10);
    const mm = parseInt(mmStr ?? "0", 10);
    return hh * 60 + mm;
  }
  
  /**
   * Сливает перекрывающиеся интервалы.
   * intervals: Array<[startMin, endMin]>
   * Возвращает массив слитых интервалов.
   */
  function mergeIntervals(intervals: Array<[number, number]>): Array<[number, number]> {
    if (!intervals.length) return [];
    // сортируем по началу
    intervals.sort((a, b) => a[0] - b[0]);
    const merged: Array<[number, number]> = [];
    let [curStart, curEnd] = intervals[0];
    for (let i = 1; i < intervals.length; i++) {
      const [s, e] = intervals[i];
      if (s <= curEnd) {
        // пересекаются или стыкуются
        curEnd = Math.max(curEnd, e);
      } else {
        merged.push([curStart, curEnd]);
        curStart = s;
        curEnd = e;
      }
    }
    merged.push([curStart, curEnd]);
    return merged;
  }
  
  /**
   * Основная функция: принимает слоты (SimpleSlot[]) и возвращает суммарное время в часах.
   * Результат — number. Если целое число часов, вернёт целое (например 11).
   * Если дробное — с точностью до 2 знаков (например 3.5 или 2.75 -> 2.75).
   */
  export function calculateDayHours(slots: SimpleSlot[]): number {
    if (!slots || slots.length === 0) return 0;
  
    const intervals: Array<[number, number]> = [];
  
    for (const slot of slots) {
      // защита — если формата нет, пропустить
      if (!slot?.startTime || !slot?.endTime) continue;
  
      let s = timeToMinutes(slot.startTime);
      let e = timeToMinutes(slot.endTime);
  
      // Если конец раньше начала — считаем переход через полночь (увеличим конец)
      if (e <= s) {
        e += 24 * 60;
      }
  
      intervals.push([s, e]);
    }
  
    if (!intervals.length) return 0;
  
    const merged = mergeIntervals(intervals);
  
    let totalMinutes = 0;
    for (const [s, e] of merged) {
      totalMinutes += (e - s);
    }
  
    const hours = totalMinutes / 60;
    // округлим до 2 знаков, но избавимся от лишних .00
    const rounded = Math.round(hours * 100) / 100;
    return rounded;
  }
  