// src/app/all-slots/page.tsx

"use client";

import { useState, useEffect } from "react";
import { format } from "date-fns";
import { ru } from "date-fns/locale";
import { getCalendarWeeks } from "@/lib/dateUtils";
import { TimeSlot, Day } from "@/types/shifts";
import TimeSlotComponent from "@/components/TimeSlotComponent";

export default function AllSlotsPage() {
    const [currentDayIndex, setCurrentDayIndex] = useState(0); // 0 = понедельник, 6 = воскресенье
    const [weekDays, setWeekDays] = useState<Day[]>([]);

    useEffect(() => {
        // Получаем текущую неделю
        const { mainWeek } = getCalendarWeeks(new Date());
        setWeekDays(mainWeek);
    }, []);

    const handlePrevDay = () => {
        if (currentDayIndex > 0) {
            setCurrentDayIndex(currentDayIndex - 1);
        }
    };

    const handleNextDay = () => {
        if (currentDayIndex < 6) {
            setCurrentDayIndex(currentDayIndex + 1);
        }
    };

    // Форматируем текущий день
    const getCurrentDayText = () => {
        if (weekDays.length === 0) return "";
        
        const currentDay = weekDays[currentDayIndex];
        const dayName = format(currentDay.date, "EEEE", { locale: ru });
        const formattedDate = format(currentDay.date, "d MMMM", { locale: ru });
        
        // Делаем первую букву заглавной
        const capitalizedDayName = dayName.charAt(0).toUpperCase() + dayName.slice(1);
        
        return `${capitalizedDayName}, ${formattedDate}`;
    };

    const isPrevDisabled = currentDayIndex === 0;
    const isNextDisabled = currentDayIndex === 6;

    // Временные данные для демонстрации
    const mockSlots: TimeSlot[] = [
        { id: 1, startTime: "09:00", endTime: "12:00", status: "taken", user_id: 5, userName: "egor" },
        { id: 2, startTime: "14:00", endTime: "18:00", status: "taken", user_id: 5, userName: "egor" },
    ];

    const currentDay: Day = weekDays[currentDayIndex] || { 
        date: new Date(), 
        formattedDate: "", 
        isToday: false, 
        slots: [] 
    };

    return (
        <div className="min-h-screen bg-black text-white">
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

            {/* Карточка пользователя egor */}
            <div className="mt-[50px] mx-[19px] rounded-[20px] bg-[#FFEA00]/80 min-h-[100px] pt-[10px] pl-[15px] pr-[15px] pb-[15px]">
                {/* Никнейм */}
                <p className="text-black font-['Inter'] text-[16px] font-normal leading-normal mb-[10px]">
                    egor
                </p>
                
                {/* Слоты */}
                <div className="flex flex-col gap-[6px]">
                    {mockSlots.map((slot) => (
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
        </div>
    );
}