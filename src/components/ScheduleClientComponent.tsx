// src/components/ScheduleClientComponent.tsx
'use client';

import React, { useState, useEffect } from 'react';
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

// Вспомогательная функция для обработки ошибок
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

  // Синхронизируем состояние с пропсами и выключаем лоадер
  useEffect(() => {
    setWeekDays(initialWeekDays);
    setLoading(false);
  }, [initialWeekDays]);
  
  const navigate = (newOffset: number) => {
    if (loading) return; // Предотвращаем двойные клики
    setLoading(true);
    const params = new URLSearchParams(searchParams.toString());
    router.push(`/schedule/${newOffset}?${params.toString()}`);
  };

  const handleLogout = async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    // Используем "жесткий" редирект для надежности
    window.location.href = '/';
  };

  const refreshData = () => {
    setLoading(true);
    router.refresh();
  };

  const handleAddSlot = (day: Day) => {
    setSelectedDay(day);
    setIsModalOpen(true);
  };
  
  // --- [НАЧАЛО КАПИТАЛЬНОГО РЕМОНТА] ---
  
  const handleModalDone = async (startTime: string, endTime: string) => {
    if (!selectedDay) return;
    setLoading(true);
    try {
      // Создание нового ТИПА слота -> POST /api/shifts
      const res = await fetch('/api/shifts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date: selectedDay.date.toISOString(), startTime, endTime }),
      });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось создать слот');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false); // Выключаем лоадер при ошибке
    } finally {
      setIsModalOpen(false);
      setSelectedDay(null);
    }
  };

  const handleTakeSlot = async (day: Day, slot: TimeSlot) => {
    setLoading(true);
    try {
      // Занять КОНКРЕТНЫЙ слот -> POST /api/slots
      const res = await fetch('/api/slots', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ slotId: slot.id })
      });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось занять слот');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  const handleReleaseSlot = async (day: Day, slotId: number) => {
    setLoading(true);
    try {
      // Освободить КОНКРЕТНЫЙ слот -> DELETE /api/slots
      const res = await fetch(`/api/slots?id=${slotId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось освободить слот');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  const handleDeleteDaySlots = async (day: Day) => {
    if (!confirm(`Вы уверены, что хотите освободить все свои слоты за ${day.formattedDate}?`)) return;
    setLoading(true);
    try {
      // Освободить ВСЕ слоты за день -> DELETE /api/shifts
      const dateStr = format(day.date, 'yyyy-MM-dd');
      const res = await fetch(`/api/shifts?date=${dateStr}`, { method: 'DELETE' });
      if (!res.ok) throw new Error((await res.json()).error || 'Не удалось удалить слоты');
      refreshData();
    } catch (err) {
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };
  
  // --- [КОНЕЦ КАПИТАЛЬНОГО РЕМОНТА] ---

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
          isPrevDisabled={offset === 0 || loading} // Блокируем кнопки во время загрузки
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