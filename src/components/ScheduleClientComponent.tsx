// src/components/ScheduleClientComponent.tsx

'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Day, TimeSlot, User } from '@/types/shifts';
import Header from './Header';
import Main from './Main';
import Lower from './Lower';
import AddSlotModal from './AddSlotModal';
import { getCalendarWeeks } from '@/lib/dateUtils';
import { format } from 'date-fns';

type Props = {
  initialWeekDays: Day[];
  initialOffset: number;
  currentUser: User | null;
  isOwner: boolean;
  apiKey: string;
};

function getErrorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    if (typeof error === 'string' && error.length > 0) return error;
    return 'Произошла неизвестная ошибка.';
}

export default function ScheduleClientComponent({ 
  initialWeekDays, 
  initialOffset, 
  currentUser, 
  isOwner,
  apiKey
}: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [weekDays, setWeekDays] = useState<Day[]>(initialWeekDays);
  const [offset, setOffset] = useState(initialOffset);
  const [loading, setLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedDay, setSelectedDay] = useState<Day | null>(null);

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Функция для загрузки расписания
  const loadSchedule = async () => {
    try {
      const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
      const weekDaysTemplate = offset === 1 ? nextWeek : mainWeek;

      const startDate = format(weekDaysTemplate[0].date, "yyyy-MM-dd");
      const endDate = format(
        new Date(weekDaysTemplate[6].date.getTime() + 24 * 60 * 60 * 1000),
        "yyyy-MM-dd"
      );

      const params: Record<string, string> = { start: startDate, end: endDate };
      const viewedUserId = searchParams.get('userId');
      if (viewedUserId) params.userId = viewedUserId;

      const qs = new URLSearchParams(params).toString();

      const res = await fetch(`/api/shifts?${qs}`, {
        headers: { Authorization: `Bearer ${apiKey}` },
      });
      
      if (!res.ok) {
        console.error("Ошибка загрузки расписания");
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

      const days: Day[] = weekDaysTemplate.map((d) => ({
        ...d,
        slots: [],
      }));

      for (const r of rows) {
        const rowDateStr = String(r.shift_date);
        const dayIndex = days.findIndex(
          (wd) => format(wd.date, "yyyy-MM-dd") === rowDateStr
        );
        if (dayIndex === -1) continue;

        const [startTime = "", endTime = ""] = r.shift_code?.split("-") ?? ["", ""];

        const slot: TimeSlot = {
          id: r.id,
          startTime,
          endTime,
          status: r.status as any,
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
  };

  useEffect(() => {
    const refreshData = async () => { 
      await loadSchedule();
    };
    
    if (!isOwner) {
      pollingIntervalRef.current = setInterval(refreshData, 5000);
    }
    
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [isOwner, offset, apiKey]);

  useEffect(() => {
    setWeekDays(initialWeekDays);
    setLoading(false);
  }, [initialWeekDays]);

  const navigate = (newOffset: number) => {
    if (loading) return;
    setLoading(true);
    const params = new URLSearchParams(searchParams.toString());
    router.push(`/schedule/${newOffset}?${params.toString()}`);
  };

  const handleLogout = async () => {
    await fetch('/api/auth/logout', { 
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` }
    });
    localStorage.removeItem('apiKey');
    window.location.href = '/auth';
  };

  const refreshData = async () => {
    setLoading(true);
    await loadSchedule();
    setLoading(false);
  };

  const handleAddSlot = (day: Day) => {
    console.log('handleAddSlot called for day:', day.formattedDate);
    setSelectedDay(day);
    setIsModalOpen(true);
  };
  
  const handleModalDone = async (startTime: string, endTime: string) => {
    if (!selectedDay) return;
    setLoading(true);
    try {
      const dateStr = format(selectedDay.date, 'yyyy-MM-dd');
      console.log('Creating slot:', { date: dateStr, startTime, endTime });
      
      const res = await fetch('/api/shifts', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({ 
          date: dateStr, 
          startTime, 
          endTime,
          assignToSelf: isOwner
        }),
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Не удалось создать слот');
      }
      
      // Обновляем данные напрямую
      await refreshData();
    } catch (err) {
      console.error('Error creating slot:', err);
      alert(getErrorMessage(err)); 
    } finally {
      setIsModalOpen(false); 
      setSelectedDay(null);
      setLoading(false);
    }
  };

  const handleTakeSlot = async (day: Day, slot: TimeSlot) => {
    setLoading(true);
    try {
      const res = await fetch('/api/slots', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({ slotId: slot.id })
      });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось занять слот');
      await refreshData();
    } catch (err) {
      alert(getErrorMessage(err)); 
    } finally {
      setLoading(false);
    }
  };
  
  const handleDeleteSlot = async (day: Day, slotId: number) => {
    if (!confirm(`Вы уверены, что хотите навсегда удалить этот слот?`)) return;

    setLoading(true);
    try {
      const res = await fetch(`/api/slots?id=${slotId}`, { 
        method: 'DELETE',
        headers: { Authorization: `Bearer ${apiKey}` },
        cache: 'no-store' 
      });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось удалить слот');
      await refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteDaySlots = async (day: Day) => {
    const confirmationText = `Вы уверены, что хотите очистить день ${day.formattedDate}?\n\nБудут удалены все свободные слоты, а ваши занятые слоты станут свободными.`;
    if (!confirm(confirmationText)) return;

    setLoading(true);
    try {
      const dateStr = format(day.date, 'yyyy-MM-dd');
      const res = await fetch(`/api/shifts?date=${dateStr}`, { 
        method: 'DELETE',
        headers: { Authorization: `Bearer ${apiKey}` },
        cache: 'no-store'
      });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось очистить день');
      await refreshData();
    } catch (err) {
      alert(getErrorMessage(err)); 
    } finally {
      setLoading(false);
    }
  };

  const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
  const dateRange = offset === 0 
    ? `${format(mainWeek[0].date, 'd MMM')} - ${format(mainWeek[6].date, 'd MMM')}`
    : `${format(nextWeek[0].date, 'd MMM')} - ${format(nextWeek[6].date, 'd MMM')}`;

  console.log('ScheduleClientComponent - isOwner:', isOwner);

  return (
    <>
      <div className="flex flex-col min-h-screen bg-black text-white pb-[70px] pt-[100px]">
        <Header 
          dateRange={dateRange}
          onPrevWeek={() => navigate(0)} 
          onNextWeek={() => navigate(1)}
          isPrevDisabled={offset === 0 || loading} 
          isNextDisabled={offset === 1 || loading}
          onLogout={handleLogout}
        />
        <Main
          weekDays={weekDays}
          onPrevWeek={() => navigate(0)} 
          onNextWeek={() => navigate(1)}
          isLoading={loading}
          currentUserId={currentUser?.id || null}
          isOwner={isOwner}
          onAddSlot={handleAddSlot}
          onTakeSlot={handleTakeSlot}
          onDeleteSlot={handleDeleteSlot}
          onDeleteDaySlots={handleDeleteDaySlots}
        />
        <Lower 
            isOwner={isOwner}
            onAddSlotClick={() => {
              const todayDay = weekDays.find(d => d.isToday) || weekDays[0];
              console.log('Lower button clicked, adding slot for:', todayDay?.formattedDate);
              handleAddSlot(todayDay);
            }}
        />
      </div>
      {isModalOpen && selectedDay && (
        <AddSlotModal
          onClose={() => {
            setIsModalOpen(false);
            setSelectedDay(null);
          }}
          onDone={handleModalDone}
        />
      )}
    </>
  );
}