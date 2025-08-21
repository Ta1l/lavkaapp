// src/app/schedule/[offset]/page.tsx
import React from "react";
import { pool } from "@/lib/db";
import { getCalendarWeeks } from "@/lib/dateUtils";
import { format } from "date-fns";
import { getUserFromSession } from "@/lib/session";
import ScheduleClientComponent from "@/components/ScheduleClientComponent";
import type { Day, ShiftStatus, TimeSlot } from "@/types/shifts"; // Импортируем ShiftStatus

interface Props {
  params: { offset: string };
  searchParams?: { userId?: string; start?: string; end?: string };
}

export default async function SchedulePage({ params, searchParams }: Props) {
  const offset = Number(params.offset ?? 0);
  const viewedUserId = searchParams?.userId ?? null;
  const startParam = searchParams?.start ?? null;
  const endParam = searchParams?.end ?? null;
  const currentUser = await getUserFromSession();

  const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
  const weekDaysTemplate = offset === 1 ? nextWeek : mainWeek;
  
  const startDate = format(weekDaysTemplate[0].date, "yyyy-MM-dd");
  const endDate = format(
    new Date(weekDaysTemplate[6].date.getTime() + 24 * 60 * 60 * 1000),
    "yyyy-MM-dd"
  );
  
  const where: string[] = [];
  const paramsSql: any[] = [];

  paramsSql.push(startDate);
  where.push(`s.shift_date >= $${paramsSql.length}`);
  paramsSql.push(endDate);
  where.push(`s.shift_date < $${paramsSql.length}`);

  if (viewedUserId) {
    paramsSql.push(parseInt(viewedUserId, 10));
    where.push(`s.user_id = $${paramsSql.length}`);
  } else if (currentUser) {
    paramsSql.push(currentUser.id);
    where.push(`(s.user_id = $${paramsSql.length} OR s.status = 'available')`);
  } else {
    where.push(`s.status = 'available'`);
  }

  const sql = `
    SELECT s.id, s.user_id, s.shift_date, s.shift_code, s.status, u.username
    FROM shifts s
    LEFT JOIN users u ON u.id = s.user_id
    WHERE ${where.join(" AND ")}
    ORDER BY s.shift_date, s.shift_code
  `;
  
  const result = await pool.query(sql, paramsSql);
  const rows = result.rows as Array<{
    id: number;
    user_id: number | null;
    shift_date: string | Date;
    shift_code: string;
    status: string;
    username?: string | null;
  }>;
  
  const weekDays: Day[] = weekDaysTemplate.map((d) => ({
    ...d,
    slots: [],
  }));
  
  for (const r of rows) {
    const rowDateStr =
      r.shift_date instanceof Date
        ? format(r.shift_date, "yyyy-MM-dd")
        : String(r.shift_date);
    const dayIndex = weekDays.findIndex(
      (wd) => format(wd.date, "yyyy-MM-dd") === rowDateStr
    );
    if (dayIndex === -1) continue;

    const [startTime = "", endTime = ""] = r.shift_code?.split("-") ?? ["", ""];
    
    // --- НАЧАЛО ИСПРАВЛЕНИЯ ---
    const slot: TimeSlot = {
      id: r.id,
      startTime,
      endTime,
      status: r.status as ShiftStatus, // Явно указываем TypeScript, что это ShiftStatus
      user_id: r.user_id,
      userName: r.username ?? null,
    };
    // --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    weekDays[dayIndex].slots.push(slot);
  }

  weekDays.forEach((d) => {
    d.slots.sort((a, b) => (a.startTime > b.startTime ? 1 : -1));
  });
  
  const isOwner =
    !!currentUser &&
    (!viewedUserId || Number(viewedUserId) === currentUser.id);
  
  return (
    <ScheduleClientComponent
      initialWeekDays={weekDays}
      initialOffset={offset}
      currentUser={currentUser}
      isOwner={isOwner}
    />
  );
}