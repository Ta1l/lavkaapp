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
};

function getErrorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    if (typeof error === 'string' && error.length > 0) return error;
    return 'Произошла неизвестная ошибка.';
}

export default function ScheduleClientComponent({ initialWeekDays, initialOffset, currentUser, isOwner }: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [weekDays, setWeekDays] = useState<Day[]>(initialWeekDays);
  const [offset, setOffset] = useState(initialOffset);
  const [loading, setLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedDay, setSelectedDay] = useState<Day | null>(null);

  // --- НАЧАЛО ИЗМЕНЕНИЙ: REAL-TIME ОБНОВЛЕНИЕ ---
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  useEffect(() => {
    // Функция для обновления данных
    const refreshData = () => {
      console.log('[ScheduleClient] Refreshing data via router.refresh()');
      router.refresh();
    };

    // Если мы просматриваем чужой профиль, запускаем поллинг
    if (!isOwner) {
      console.log('[ScheduleClient] Viewing another user profile. Starting polling for real-time updates.');
      // Устанавливаем интервал для обновления каждые 5 секунд
      pollingIntervalRef.current = setInterval(refreshData, 5000);
    }

    // Очищаем интервал при размонтировании компонента или изменении isOwner
    return () => {
      if (pollingIntervalRef.current) {
        console.log('[ScheduleClient] Clearing polling interval.');
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [isOwner, router]);
  // --- КОНЕЦ ИЗМЕНЕНИЙ: REAL-TIME ОБНОВЛЕНИЕ ---

  // Синхронизируем состояние с пропсами и выключаем лоадер
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
    console.log('[ScheduleClient] Logging out.');
    await fetch('/api/auth/logout', { method: 'POST' });
    window.location.href = '/';
  };

  const refreshData = () => {
    console.log('[ScheduleClient] Manually refreshing data.');
    setLoading(true);
    router.refresh();
  };

  const handleAddSlot = (day: Day) => {
    setSelectedDay(day);
    setIsModalOpen(true);
  };
  
  const handleModalDone = async (startTime: string, endTime: string) => {
    if (!selectedDay) return;
    setLoading(true);
    console.log(`[ScheduleClient] handleModalDone for date: ${selectedDay.date}, time: ${startTime}-${endTime}`);
    try {
      // --- НАЧАЛО ИЗМЕНЕНИЙ ---
      // Когда владелец добавляет слот, мы передаем флаг, чтобы сразу его назначить
      const res = await fetch('/api/shifts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          date: selectedDay.date.toISOString(), 
          startTime, 
          endTime,
          assignToSelf: isOwner // <<<<<<<<<<< Вот оно!
        }),
      });
      // --- КОНЕЦ ИЗМЕНЕНИЙ ---

      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось создать слот');
      console.log('[ScheduleClient] Slot created/updated successfully.');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false);
    } finally {
      setIsModalOpen(false);
      setSelectedDay(null);
    }
  };

  const handleTakeSlot = async (day: Day, slot: TimeSlot) => {
    setLoading(true);
    console.log(`[ScheduleClient] User ${currentUser?.id} taking slot ${slot.id}`);
    try {
      const res = await fetch('/api/slots', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ slotId: slot.id })
      });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось занять слот');
      console.log('[ScheduleClient] Slot taken successfully.');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  const handleReleaseSlot = async (day: Day, slotId: number) => {
    setLoading(true);
    console.log(`[ScheduleClient] User ${currentUser?.id} releasing slot ${slotId}`);
    try {
      const res = await fetch(`/api/slots?id=${slotId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось освободить слот');
      console.log('[ScheduleClient] Slot released successfully.');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  const handleDeleteDaySlots = async (day: Day) => {
    if (!confirm(`Вы уверены, что хотите освободить все свои слоты за ${day.formattedDate}?`)) return;
    setLoading(true);
    console.log(`[ScheduleClient] User ${currentUser?.id} deleting all slots for ${day.formattedDate}`);
    try {
      const dateStr = format(day.date, 'yyyy-MM-dd');
      const res = await fetch(`/api/shifts?date=${dateStr}`, { method: 'DELETE' });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось удалить слоты');
      console.log('[ScheduleClient] Day slots deleted successfully.');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
  const dateRange = offset === 0 
    ? `${format(mainWeek[0].date, 'd MMM')} - ${format(mainWeek[6].date, 'd MMM')}`
    : `${format(nextWeek[0].date, 'd MMM')} - ${format(nextWeek[6].date, 'd MMM')}`;

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
          onReleaseSlot={handleReleaseSlot}
          onDeleteDaySlots={handleDeleteDaySlots}
        />
        <Lower 
            isOwner={isOwner}
            onAddSlotClick={() => handleAddSlot(weekDays.find(d => d.isToday) || weekDays[0])}
        />
      </div>
      {isModalOpen && (
        <AddSlotModal
          onClose={() => setIsModalOpen(false)}
          onDone={handleModalDone}
        />
      )}
    </>
  );
}