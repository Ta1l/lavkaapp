// src/components/ScheduleClientComponent.tsx

'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  viewedUserId: number | null;
};

// Вынесли в константу для переиспользования и тестирования
const POLLING_INTERVAL = 5000;

// Улучшенная функция обработки ошибок
function getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        // Проверяем, не является ли это ошибкой отмены запроса
        if (error.name === 'AbortError') {
            return ''; // Не показываем ошибку для отмененных запросов
        }
        return error.message;
    }
    if (typeof error === 'string' && error.length > 0) return error;
    return 'Произошла неизвестная ошибка.';
}

// Вспомогательная функция для логирования в development
const devLog = (...args: any[]) => {
    if (process.env.NODE_ENV === 'development') {
        console.log(...args);
    }
};

export default function ScheduleClientComponent({ 
  initialWeekDays, 
  initialOffset, 
  currentUser, 
  isOwner,
  apiKey,
  viewedUserId
}: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [weekDays, setWeekDays] = useState<Day[]>(initialWeekDays);
  const [offset, setOffset] = useState(initialOffset);
  const [loading, setLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedDay, setSelectedDay] = useState<Day | null>(null);
  const [editingSlot, setEditingSlot] = useState<{day: Day, slot: TimeSlot} | null>(null);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);

  // Refs для хранения состояний между рендерами
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef<boolean>(true);

  // 🔧 ИСПРАВЛЕНИЕ: useCallback для мемоизации функции loadSchedule
  // Это предотвращает пересоздание функции и решает проблему stale closure
  const loadSchedule = useCallback(async (signal?: AbortSignal) => {
    try {
      devLog('🔄 Loading schedule...');
      
      const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
      const weekDaysTemplate = offset === 1 ? nextWeek : mainWeek;

      const startDate = format(weekDaysTemplate[0].date, "yyyy-MM-dd");
      const endDate = format(
        new Date(weekDaysTemplate[6].date.getTime() + 24 * 60 * 60 * 1000),
        "yyyy-MM-dd"
      );

      devLog('📅 Date range:', startDate, 'to', endDate);

      const params: Record<string, string> = { start: startDate, end: endDate };
      
      if (viewedUserId) {
        params.userId = String(viewedUserId);
        devLog('👀 Viewing user:', viewedUserId);
      }

      const qs = new URLSearchParams(params).toString();
      const url = `/api/shifts?${qs}`;
      devLog('🌐 Fetching:', url);

      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${apiKey}` },
        signal, // 🔧 ИСПРАВЛЕНИЕ: Передаем signal для возможности отмены запроса
      });
      
      // 🔧 ИСПРАВЛЕНИЕ: Проверяем, не был ли компонент размонтирован
      if (!isMountedRef.current) {
        return;
      }

      if (!res.ok) {
        console.error("❌ Ошибка загрузки расписания:", res.status);
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
      
      devLog('📊 Loaded shifts:', rows.length, 'items');

      const days: Day[] = weekDaysTemplate.map((d) => ({
        ...d,
        slots: [],
      }));

      for (const r of rows) {
        const rowDateStr = String(r.shift_date).split('T')[0];
        
        const dayIndex = days.findIndex(
          (wd) => format(wd.date, "yyyy-MM-dd") === rowDateStr
        );
        
        if (dayIndex === -1) {
          devLog('⚠️ Day not found for date:', rowDateStr);
          continue;
        }

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

      devLog('📊 Final days with slots:', days.map(d => ({
        date: format(d.date, 'yyyy-MM-dd'),
        formattedDate: d.formattedDate,
        slotsCount: d.slots.length
      })));

      // 🔧 ИСПРАВЛЕНИЕ: Финальная проверка перед обновлением состояния
      if (isMountedRef.current && !signal?.aborted) {
        setWeekDays(days);
      }
    } catch (error: any) {
      // 🔧 ИСПРАВЛЕНИЕ: Не логируем ошибки отмененных запросов
      if (error?.name !== 'AbortError') {
        console.error("❌ Ошибка загрузки расписания:", error);
      }
    }
  }, [offset, apiKey, viewedUserId]); // Правильные зависимости

  // 🔧 ИСПРАВЛЕНИЕ: Основной эффект для загрузки данных с AbortController
  useEffect(() => {
    // Отменяем предыдущий запрос, если он существует
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Создаем новый AbortController для этого запроса
    abortControllerRef.current = new AbortController();
    
    // Загружаем данные с возможностью отмены
    loadSchedule(abortControllerRef.current.signal);

    // Cleanup функция
    return () => {
      // Отменяем запрос при размонтировании или изменении зависимостей
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, [loadSchedule]); // loadSchedule теперь стабильна благодаря useCallback

  // 🔧 ИСПРАВЛЕНИЕ: Отдельный эффект для polling с правильными зависимостями
  useEffect(() => {
    // Функция для polling, которая проверяет mounted состояние
    const pollData = () => {
      if (isMountedRef.current) {
        // Создаем отдельный AbortController для polling запроса
        const pollAbortController = new AbortController();
        loadSchedule(pollAbortController.signal);
        
        // Сохраняем ссылку для возможной отмены
        return pollAbortController;
      }
      return null;
    };

    let pollAbortController: AbortController | null = null;

    if (!isOwner) {
      // Запускаем polling только для не-владельцев
      pollingIntervalRef.current = setInterval(() => {
        // Отменяем предыдущий polling запрос, если он еще выполняется
        if (pollAbortController) {
          pollAbortController.abort();
        }
        pollAbortController = pollData();
      }, POLLING_INTERVAL);

      devLog('📡 Polling started');
    }

    // Cleanup функция
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
        devLog('📡 Polling stopped');
      }
      if (pollAbortController) {
        pollAbortController.abort();
      }
    };
  }, [isOwner, loadSchedule]); // Правильные зависимости

  // 🔧 ИСПРАВЛЕНИЕ: Эффект для отслеживания mounted состояния
  useEffect(() => {
    isMountedRef.current = true;
    
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // 🔧 ИСПРАВЛЕНИЕ: Предотвращаем race condition при навигации
  const navigate = useCallback((newOffset: number) => {
    if (loading) return;
    
    // Отменяем текущие запросы перед навигацией
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    setLoading(true);
    setOffset(newOffset); // Обновляем offset, что триггерит useEffect
    
    const params = new URLSearchParams(searchParams.toString());
    router.push(`/schedule/${newOffset}?${params.toString()}`);
  }, [loading, router, searchParams]);

  // 🔧 ИСПРАВЛЕНИЕ: Обновленный handleLogout с проверкой mounted
  const handleLogout = async () => {
    try {
      await fetch('/api/auth/logout', { 
        method: 'POST',
        headers: { Authorization: `Bearer ${apiKey}` }
      });
      localStorage.removeItem('apiKey');
      
      if (isMountedRef.current) {
        window.location.href = '/auth';
      }
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  // 🔧 ИСПРАВЛЕНИЕ: Улучшенная функция обновления с AbortController
  const refreshData = useCallback(async () => {
    if (!isMountedRef.current) return;
    
    setLoading(true);
    
    // Создаем новый AbortController для refresh запроса
    const refreshAbortController = new AbortController();
    
    try {
      await loadSchedule(refreshAbortController.signal);
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
    
    return () => {
      refreshAbortController.abort();
    };
  }, [loadSchedule]);

  // Handlers остаются похожими, но с добавлением проверок mounted
  const handleAddSlot = useCallback((day: Day) => {
    devLog('handleAddSlot called for day:', day.formattedDate);
    setSelectedDay(day);
    setIsModalOpen(true);
  }, []);
  
  const handleEditSlot = useCallback((day: Day, slot: TimeSlot) => {
    devLog('handleEditSlot called for slot:', slot);
    setEditingSlot({ day, slot });
    setIsEditModalOpen(true);
  }, []);
  
  // 🔧 ИСПРАВЛЕНИЕ: handleModalDone с проверками mounted состояния
  const handleModalDone = async (startTime: string, endTime: string) => {
    if (!isMountedRef.current) return;
    
    setLoading(true);
    
    try {
      if (editingSlot) {
        devLog('📝 Editing slot:', editingSlot.slot.id);
        
        const res = await fetch('/api/slots', {
          method: 'PATCH',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({ 
            slotId: editingSlot.slot.id,
            startTime, 
            endTime 
          }),
        });
        
        const responseText = await res.text();
        devLog('📥 Edit response status:', res.status);
        
        let responseData;
        try {
          responseData = JSON.parse(responseText);
        } catch (e) {
          console.error('❌ Failed to parse edit response:', e);
          throw new Error('Invalid response from server');
        }
        
        if (!res.ok) {
          throw new Error(responseData.error || 'Не удалось обновить слот');
        }
        
        devLog('✅ Slot updated successfully');
      } else if (selectedDay) {
        devLog('➕ Creating new slot');
        
        const dateStr = format(selectedDay.date, 'yyyy-MM-dd');
        const requestBody = { 
          date: dateStr, 
          startTime, 
          endTime,
          assignToSelf: isOwner
        };
        
        devLog('📤 Creating slot with data:', requestBody);
        
        const res = await fetch('/api/shifts', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify(requestBody),
        });
        
        const responseText = await res.text();
        devLog('📥 Response status:', res.status);
        
        let responseData;
        try {
          responseData = JSON.parse(responseText);
        } catch (e) {
          console.error('❌ Failed to parse response:', e);
          throw new Error('Invalid response from server');
        }
        
        if (!res.ok) {
          throw new Error(responseData.error || 'Не удалось создать слот');
        }
        
        devLog('✅ Slot created successfully');
      }
      
      // Проверяем mounted перед обновлением
      if (isMountedRef.current) {
        await refreshData();
        devLog('✅ Data refreshed');
      }
      
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      if (errorMessage && isMountedRef.current) {
        alert(errorMessage);
      }
    } finally {
      if (isMountedRef.current) {
        setIsModalOpen(false);
        setIsEditModalOpen(false);
        setSelectedDay(null);
        setEditingSlot(null);
        setLoading(false);
      }
    }
  };

  // 🔧 ИСПРАВЛЕНИЕ: Остальные handlers с проверками mounted
  const handleTakeSlot = async (day: Day, slot: TimeSlot) => {
    if (!isMountedRef.current) return;
    
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
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Не удалось занять слот');
      }
      
      if (isMountedRef.current) {
        await refreshData();
      }
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      if (errorMessage && isMountedRef.current) {
        alert(errorMessage);
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  };
  
  const handleDeleteSlot = async (day: Day, slotId: number) => {
    if (!confirm(`Вы уверены, что хотите навсегда удалить этот слот?`)) return;
    if (!isMountedRef.current) return;

    setLoading(true);
    try {
      const res = await fetch(`/api/slots?id=${slotId}`, { 
        method: 'DELETE',
        headers: { Authorization: `Bearer ${apiKey}` },
        cache: 'no-store' 
      });
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Не удалось удалить слот');
      }
      
      if (isMountedRef.current) {
        await refreshData();
      }
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      if (errorMessage && isMountedRef.current) {
        alert(errorMessage);
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  };

  const handleDeleteDaySlots = async (day: Day) => {
    const confirmationText = `Вы уверены, что хотите очистить день ${day.formattedDate}?\n\nБудут удалены все свободные слоты, а ваши занятые слоты станут свободными.`;
    if (!confirm(confirmationText)) return;
    if (!isMountedRef.current) return;

    setLoading(true);
    try {
      const dateStr = format(day.date, 'yyyy-MM-dd');
      const res = await fetch(`/api/shifts?date=${dateStr}`, { 
        method: 'DELETE',
        headers: { Authorization: `Bearer ${apiKey}` },
        cache: 'no-store'
      });
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Не удалось очистить день');
      }
      
      if (isMountedRef.current) {
        await refreshData();
      }
    } catch (err) {
      const errorMessage = getErrorMessage(err);
      if (errorMessage && isMountedRef.current) {
        alert(errorMessage);
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  };

  const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
  const dateRange = offset === 0 
    ? `${format(mainWeek[0].date, 'd MMM')} - ${format(mainWeek[6].date, 'd MMM')}`
    : `${format(nextWeek[0].date, 'd MMM')} - ${format(nextWeek[6].date, 'd MMM')}`;

  devLog('ScheduleClientComponent - isOwner:', isOwner, 'viewedUserId:', viewedUserId);

  // Эффект для синхронизации с initial данными (на случай изменения извне)
  useEffect(() => {
    if (initialWeekDays && initialWeekDays.length > 0) {
      setWeekDays(initialWeekDays);
      setLoading(false);
    }
  }, [initialWeekDays]);

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
          onEditSlot={handleEditSlot}
          onTakeSlot={handleTakeSlot}
          onDeleteSlot={handleDeleteSlot}
          onDeleteDaySlots={handleDeleteDaySlots}
        />
        <Lower 
            isOwner={isOwner}
            onAddSlotClick={() => {
              const todayDay = weekDays.find(d => d.isToday) || weekDays[0];
              devLog('Lower button clicked, adding slot for:', todayDay?.formattedDate);
              handleAddSlot(todayDay);
            }}
        />
      </div>
      
      {/* Модальное окно для создания слота */}
      {isModalOpen && selectedDay && (
        <AddSlotModal
          onClose={() => {
            setIsModalOpen(false);
            setSelectedDay(null);
          }}
          onDone={handleModalDone}
        />
      )}
      
      {/* Модальное окно для редактирования слота */}
      {isEditModalOpen && editingSlot && (
        <AddSlotModal
          onClose={() => {
            setIsEditModalOpen(false);
            setEditingSlot(null);
          }}
          onDone={handleModalDone}
          initialStartTime={editingSlot.slot.startTime}
          initialEndTime={editingSlot.slot.endTime}
        />
      )}
    </>
  );
}