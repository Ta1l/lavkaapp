// src/app/all-slots/page.tsx

"use client";

import { useState, useEffect } from "react";
import { format } from "date-fns";
import { ru } from "date-fns/locale";
import { getCalendarWeeks } from "@/lib/dateUtils";
import { TimeSlot, Day } from "@/types/shifts";
import TimeSlotComponent from "@/components/TimeSlotComponent";
import { useSwipeable, SwipeableHandlers } from "react-swipeable";

interface UserSlots {
    userId: number;
    userName: string;
    slots: TimeSlot[];
}

export default function AllSlotsPage() {
    const [currentDayIndex, setCurrentDayIndex] = useState(0);
    const [weekOffset, setWeekOffset] = useState(0); // 0 = текущая неделя, 1 = следующая
    const [weekDays, setWeekDays] = useState<Day[]>([]);
    const [userSlots, setUserSlots] = useState<UserSlots[]>([]);
    const [loading, setLoading] = useState(false);
    const [apiKey, setApiKey] = useState<string>("");

    useEffect(() => {
        // Получаем недели
        const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
        const selectedWeek = weekOffset === 0 ? mainWeek : nextWeek;
        setWeekDays(selectedWeek);
        
        // Получаем API ключ из localStorage
        const storedApiKey = localStorage.getItem("apiKey");
        if (storedApiKey) {
            setApiKey(storedApiKey);
        }
    }, [weekOffset]);

    useEffect(() => {
        if (weekDays.length > 0 && apiKey) {
            loadSlotsForDay();
        }
    }, [currentDayIndex, weekDays, apiKey]);

    const loadSlotsForDay = async () => {
        if (!weekDays[currentDayIndex]) return;

        setLoading(true);
        try {
            const currentDate = format(weekDays[currentDayIndex].date, "yyyy-MM-dd");
            const nextDate = format(new Date(weekDays[currentDayIndex].date.getTime() + 24 * 60 * 60 * 1000), "yyyy-MM-dd");
            
            const res = await fetch(`/api/shifts?start=${currentDate}&end=${nextDate}&allUsers=true`, {
                headers: { 
                    Authorization: `Bearer ${apiKey}` 
                },
            });
            
            if (!res.ok) {
                console.error("Ошибка загрузки слотов:", res.status);
                return;
            }

            const data = await res.json();
            console.log("Загруженные слоты:", data);
            
            // Группируем слоты по пользователям
            const groupedSlots = new Map<number, UserSlots>();
            
            data.forEach((shift: any) => {
                if (shift.user_id) {
                    const [startTime, endTime] = shift.shift_code?.split("-") || ["", ""];
                    
                    const slot: TimeSlot = {
                        id: shift.id,
                        startTime: startTime.trim(),
                        endTime: endTime.trim(),
                        status: shift.status as any,
                        user_id: shift.user_id,
                        userName: shift.username || shift.full_name || `user${shift.user_id}`
                    };

                    if (!groupedSlots.has(shift.user_id)) {
                        groupedSlots.set(shift.user_id, {
                            userId: shift.user_id,
                            userName: shift.username || shift.full_name || `user${shift.user_id}`,
                            slots: []
                        });
                    }
                    
                    groupedSlots.get(shift.user_id)!.slots.push(slot);
                }
            });
            
            // Преобразуем Map в массив и сортируем по userId
            const sortedUserSlots = Array.from(groupedSlots.values())
                .sort((a, b) => a.userId - b.userId);
            
            // Сортируем слоты внутри каждого пользователя по времени
            sortedUserSlots.forEach(user => {
                user.slots.sort((a, b) => a.startTime.localeCompare(b.startTime));
            });
            
            console.log("Сгруппированные слоты:", sortedUserSlots);
            setUserSlots(sortedUserSlots);
        } catch (error) {
            console.error("Ошибка загрузки данных:", error);
        } finally {
            setLoading(false);
        }
    };

    const handlePrevDay = () => {
        if (currentDayIndex > 0) {
            // Переход на предыдущий день в текущей неделе
            setCurrentDayIndex(currentDayIndex - 1);
        } else if (weekOffset > 0) {
            // Переход на предыдущую неделю, воскресенье
            setWeekOffset(0);
            setCurrentDayIndex(6);
        }
    };

    const handleNextDay = () => {
        if (currentDayIndex < 6) {
            // Переход на следующий день в текущей неделе
            setCurrentDayIndex(currentDayIndex + 1);
        } else if (weekOffset === 0) {
            // Переход на следующую неделю, понедельник
            setWeekOffset(1);
            setCurrentDayIndex(0);
        }
    };

    // Swipe handlers
    const swipeHandlers: SwipeableHandlers = useSwipeable({
        onSwipedLeft: () => {
            if (!isNextDisabled) {
                handleNextDay();
            }
        },
        onSwipedRight: () => {
            if (!isPrevDisabled) {
                handlePrevDay();
            }
        },
        preventScrollOnSwipe: true,
        trackMouse: false,
        delta: 30, // минимальное расстояние свайпа
    });

    const getCurrentDayText = () => {
        if (weekDays.length === 0) return "";
        
        const currentDay = weekDays[currentDayIndex];
        const dayName = format(currentDay.date, "EEEE", { locale: ru });
        const formattedDate = format(currentDay.date, "d MMMM", { locale: ru });
        
        const capitalizedDayName = dayName.charAt(0).toUpperCase() + dayName.slice(1);
        
        return `${capitalizedDayName}, ${formattedDate}`;
    };

    // Определяем, можно ли перейти назад/вперед
    const isPrevDisabled = currentDayIndex === 0 && weekOffset === 0;
    const isNextDisabled = currentDayIndex === 6 && weekOffset === 1;

    const currentDay: Day = weekDays[currentDayIndex] || { 
        date: new Date(), 
        formattedDate: "", 
        isToday: false, 
        slots: [] 
    };

    return (
        <div className="min-h-screen bg-black text-white" {...swipeHandlers}>
            {/* Навигация по дням */}
            <div className="relative w-full pt-[43px]">
                {/* Кнопка влево */}
                <button
                    onClick={handlePrevDay}
                    disabled={isPrevDisabled}
                    className={`absolute left-[14px] top-[43px] w-[25px] h-[25px] transition-all ${
                        isPrevDisabled 
                            ? 'opacity-50 cursor-not-allowed' 
                            : 'active:scale-95 cursor-pointer hover:opacity-80'
                    }`}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 25 25" fill="none">
                        <circle cx="12.5" cy="12.5" r="12.5" fill="#353333"/>
                        <path d="M12.2703 12L16 15.8333L14.8649 17L10 12L14.8649 7L16 8.16667L12.2703 12Z" fill="white"/>
                    </svg>
                </button>

                {/* Поле с днем недели */}
                <div className="mx-auto w-[246px] h-[27px] rounded-[20px] bg-[#353333] flex items-center justify-center">
                    <span className="text-white text-[14px] font-normal font-['Jura'] leading-normal">
                        {getCurrentDayText()}
                    </span>
                </div>

                {/* Кнопка вправо */}
                <button
                    onClick={handleNextDay}
                    disabled={isNextDisabled}
                    className={`absolute right-[14px] top-[43px] w-[25px] h-[25px] transition-all ${
                        isNextDisabled 
                            ? 'opacity-50 cursor-not-allowed' 
                            : 'active:scale-95 cursor-pointer hover:opacity-80'
                    }`}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 25 25" fill="none">
                        <circle cx="12.5" cy="12.5" r="12.5" fill="#353333"/>
                        <path d="M13.7297 12L10 8.16667L11.1351 7L16 12L11.1351 17L10 15.8333L13.7297 12Z" fill="white"/>
                    </svg>
                </button>
            </div>

            {/* Карточки пользователей со слотами */}
            <div className="mt-[50px] px-[19px] pb-[20px]">
                {loading ? (
                    <div className="text-center text-gray-400 mt-20">Загрузка...</div>
                ) : userSlots.length > 0 ? (
                    userSlots.map((user, index) => (
                        <div 
                            key={user.userId}
                            className={`rounded-[20px] bg-[#FFEA00]/90 min-h-[100px] pt-[10px] pl-[15px] pr-[15px] pb-[15px] ${
                                index < userSlots.length - 1 ? 'mb-[15px]' : ''
                            }`}
                        >
                            {/* Никнейм */}
                            <p className="text-black font-['Jura'] text-[16px] font-normal leading-normal mb-[10px]">
                                {user.userName}
                            </p>
                            
                            {/* Слоты */}
                            <div className="flex flex-col gap-[6px]">
                                {user.slots.map((slot) => (
                                    <TimeSlotComponent
                                        key={slot.id}
                                        slot={slot}
                                        day={currentDay}
                                        onTakeSlot={() => {}}
                                        onDeleteSlot={() => {}}
                                        currentUserId={null}
                                        isOwner={false}
                                    />
                                ))}
                            </div>
                        </div>
                    ))
                ) : (
                    <div className="text-center text-gray-400 mt-20">
                        Нет слотов на {getCurrentDayText()}
                    </div>
                )}
            </div>
        </div>
    );
}