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
import { format, addDays } from 'date-fns';

type Props = {
  initialWeekDays: Day[];
  initialOffset: number;
  currentUser: User | null;
  isOwner: boolean;
};

// Вспомогательная функция для обработки ошибок
function getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    if (typeof error === 'string' && error.length > 0) {
        return error;
    }
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

  useEffect(() => {
    setWeekDays(initialWeekDays);
  }, [initialWeekDays]);
  
  const navigate = (newOffset: number) => {
    setLoading(true);
    const params = new URLSearchParams(searchParams.toString());
    router.push(`/schedule/${newOffset}?${params.toString()}`);
  };

  const handleLogout = async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    router.push('/');
    router.refresh();
  };

  const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
  const dateRange = offset === 0 
    ? `${format(mainWeek[0].date, 'd MMM')} - ${format(mainWeek[6].date, 'd MMM')}`
    : `${format(nextWeek[0].date, 'd MMM')} - ${format(nextWeek[6].date, 'd MMM')}`;

  const refreshData = () => {
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
    try {
      const payload = {
        date: selectedDay.date.toISOString(),
        startTime,
        endTime,
      };
      const res = await fetch('/api/shifts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.error || 'Не удалось создать слот');
      }
      
      refreshData();

    } catch (err) {
      console.error('[Client] handleModalDone error:', err);
      // --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
      alert(getErrorMessage(err));
    } finally {
      setIsModalOpen(false);
      setSelectedDay(null);
    }
  };

  const handleTakeSlot = async (day: Day, slot: TimeSlot) => {
    setLoading(true);
    try {
      const payload = {
        date: day.date.toISOString(),
        startTime: slot.startTime,
        endTime: slot.endTime,
      };
      const res = await fetch('/api/shifts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
      });
      if (!res.ok) {
          const body = await res.json();
          throw new Error(body.error || 'Не удалось занять слот');
      }
      refreshData();
    } catch (err) {
      console.error('[Client] handleTakeSlot error:', err);
      // --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  const handleReleaseSlot = async (day: Day, slotId: number) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/shifts?id=${slotId}`, {
        method: 'DELETE',
      });

      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.error || 'Не удалось освободить слот');
      }
      
      refreshData();

    } catch (err) {
      console.error('[Client] handleReleaseSlot error:', err);
      // --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  const handleDeleteDaySlots = async (day: Day) => {
    if (!confirm(`Вы уверены, что хотите удалить все слоты за ${day.formattedDate}?`)) {
      return;
    }
    setLoading(true);
    try {
      const dateStr = format(day.date, 'yyyy-MM-dd');
      const res = await fetch(`/api/shifts?date=${dateStr}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.error || 'Не удалось удалить слоты');
      }
      refreshData();
    } catch (err) {
      console.error('[Client] handleDeleteDaySlots error:', err);
      // --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
      alert(getErrorMessage(err));
      setLoading(false);
    }
  };

  useEffect(() => {
    setLoading(false);
  }, [weekDays]);

  return (
    <>
      <div className="flex flex-col min-h-screen bg-black text-white pb-[70px] pt-[100px]">
        <Header 
          dateRange={dateRange}
          onPrevWeek={() => navigate(0)}
          onNextWeek={() => navigate(1)}
          isPrevDisabled={offset === 0}
          isNextDisabled={offset === 1}
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