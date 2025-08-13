// lib/dateUtils.ts

import { startOfWeek, addDays, format, isSameDay, addWeeks } from 'date-fns';
// --- ОКОНЧАТЕЛЬНОЕ ИСПРАВЛЕНИЕ: ПОЛНЫЙ ПУТЬ К ФАЙЛУ ---
import ru from 'date-fns/locale/ru/index.js';
import { Day } from '@/types/shifts'; 

export const getMonday = (date: Date): Date => {
  return startOfWeek(date, { weekStartsOn: 1 });
};

export const generateWeekDays = (startDate: Date, today: Date): Day[] => {
  const weekDays: Day[] = [];
  for (let i = 0; i < 7; i++) {
    const date = addDays(startDate, i);
    weekDays.push({
      date,
      formattedDate: format(date, 'd MMMM, EEEE', { locale: ru }),
      isToday: isSameDay(date, today),
      slots: [],
    });
  }
  return weekDays;
};

export const getCalendarWeeks = (currentDate: Date = new Date()) => {
  const mainWeekMonday = getMonday(currentDate);
  const nextWeekMonday = addWeeks(mainWeekMonday, 1);

  const mainWeek = generateWeekDays(mainWeekMonday, currentDate);
  const nextWeek = generateWeekDays(nextWeekMonday, currentDate);

  return { mainWeek, nextWeek };
};