// src/app/schedule/[offset]/page.tsx

"use client";

import React, { useEffect, useState } from "react";
import { format } from "date-fns";
import { autoLogin } from "@/utils/autoLogin";
import ScheduleClientComponent from "@/components/ScheduleClientComponent";
import type { Day, ShiftStatus, TimeSlot, User } from "@/types/shifts";
import { getCalendarWeeks } from "@/lib/dateUtils";

interface Props {
  params: { offset: string };
  searchParams?: { userId?: string; start?: string; end?: string };
}

export default function SchedulePage({ params, searchParams }: Props) {
  const offset = Number(params.offset ?? 0);
  const viewedUserId = searchParams?.userId ? Number(searchParams.userId) : null;

  const [apiKey, setApiKey] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [weekDays, setWeekDays] = useState<Day[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function init() {
      try {
        const key = await autoLogin();
        if (key) {
          setApiKey(key);
          console.log("✅ Авторизация через Telegram успешна");

          const res = await fetch("/api/auth/me", {
            headers: { Authorization: `Bearer ${key}` },
          });
          if (res.ok) {
            const user = await res.json();
            setCurrentUser(user);
            console.log("✅ Текущий пользователь:", user);
            await loadSchedule(key, user); 
          } else {
            await loadSchedule(key, null);
          }
        } else {
          console.log("❌ Авто-логин не сработал, нужен ручной вход");
        }
      } catch (error) {
        console.error("Ошибка инициализации:", error);
      } finally {
        setLoading(false);
      }
    }

    init();
  }, [offset, viewedUserId]);

  async function loadSchedule(key: string, user: User | null) {
    try {
      const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
      const weekDaysTemplate = offset === 1 ? nextWeek : mainWeek;

      const startDate = format(weekDaysTemplate[0].date, "yyyy-MM-dd");
      const endDate = format(
        new Date(weekDaysTemplate[6].date.getTime() + 24 * 60 * 60 * 1000),
        "yyyy-MM-dd"
      );

      const params: Record<string, string> = { start: startDate, end: endDate };
      
      // Если есть viewedUserId, используем его для запроса
      if (viewedUserId) {
        params.userId = String(viewedUserId);
        console.log("👀 Загружаем расписание для пользователя:", viewedUserId);
      }

      const qs = new URLSearchParams(params).toString();
      const url = `/api/shifts?${qs}`;
      console.log("📡 Запрос расписания:", url);

      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${key}` },
      });
      if (!res.ok) {
        console.error("Ошибка загрузки расписания");
        setWeekDays(weekDaysTemplate.map(d => ({ ...d, slots: [] })));
        return;
      }

      const rows: Array<{
        id: number;
        user_id: number | null;
        shift_date: string;
        shift_code: string;
        status: string;
        username?: string | null;
      }> = await res.json();

      console.log("📊 Загружено слотов:", rows.length);

      const days: Day[] = weekDaysTemplate.map((d) => ({
        ...d,
        slots: [],
      }));
      
      for (const r of rows) {
        const rowDateStr = r.shift_date.split('T')[0];
        const dayIndex = days.findIndex(
          (wd) => format(wd.date, "yyyy-MM-dd") === rowDateStr
        );
        if (dayIndex === -1) continue;

        const [startTime = "", endTime = ""] = r.shift_code?.split("-") ?? ["", ""];

        const slot: TimeSlot = {
          id: r.id,
          startTime,
          endTime,
          status: r.status as ShiftStatus,
          user_id: r.user_id,
          userName: r.username ?? null,
        };

        days[dayIndex].slots.push(slot);
      }

      days.forEach((d) => {
        d.slots.sort((a, b) => (a.startTime > b.startTime ? 1 : -1));
      });

      setWeekDays(days);
    } catch (error) {
      console.error("Ошибка загрузки расписания:", error);
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black">
        <div className="text-white">⏳ Загрузка...</div>
      </div>
    );
  }

  if (!apiKey) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-black">
        <div className="text-white">🔑 Пожалуйста, войдите через логин/пароль</div>
      </div>
    );
  }

  // Определяем, является ли текущий пользователь владельцем просматриваемого расписания
  const isOwner = !!currentUser && (!viewedUserId || viewedUserId === currentUser.id);

  console.log("isOwner:", isOwner, "currentUser:", currentUser?.id, "viewedUserId:", viewedUserId);
  
  return (
    <ScheduleClientComponent
      initialWeekDays={weekDays}
      initialOffset={offset}
      currentUser={currentUser}
      isOwner={isOwner}
      apiKey={apiKey}
      viewedUserId={viewedUserId}
    />
  );
}