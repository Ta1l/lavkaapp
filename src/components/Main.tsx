// src/components/Main.tsx

"use client";

import { useSwipeable, SwipeableHandlers } from "react-swipeable";
import React from "react";
import TimeSlotComponent from "./TimeSlotComponent";
import { Day, TimeSlot } from "@/types/shifts";
import clsx from "clsx";

interface MainProps {
    weekDays: Day[];
    onNextWeek: () => void;
    onPrevWeek: () => void;
    isLoading?: boolean;
    currentUserId: number | null;
    onTakeSlot: (day: Day, slot: TimeSlot) => void;
    onDeleteSlot: (day: Day, slotId: number) => void; // <--- ИЗМЕНЕНО НАЗВАНИЕ
    onAddSlot: (day: Day) => void;
    onDeleteDaySlots: (day: Day) => void;
    isOwner: boolean;
}

function DayRow({
    day,
    currentUserId,
    onTakeSlot,
    onDeleteSlot, // <--- ИЗМЕНЕНО НАЗВАНИЕ
    onAddSlot,
    onDeleteDaySlots,
    isOwner,
}: {
    day: Day;
    currentUserId: number | null;
    onTakeSlot: (day: Day, slot: TimeSlot) => void;
    onDeleteSlot: (day: Day, slotId: number) => void; // <--- ИЗМЕНЕНО НАЗВАНИЕ
    onAddSlot: (day: Day) => void;
    onDeleteDaySlots: (day: Day) => void;
    isOwner: boolean;
}) {
    const getHeight = (slotsCount: number) => {
        switch (slotsCount) {
            case 0: return 'h-[50px]';
            case 1: return 'h-[75px]';
            case 2: return 'h-[116px]';
            case 3: return 'h-[157px]';
            default: return 'h-[50px]';
        }
    };
    const slots = day.slots || [];
    const slotsCount = slots.length;
    return (
        <div className={`w-full ${getHeight(slotsCount)} relative transition-all duration-300`}>
            <p className={clsx("absolute top-0 left-[9px] text-[14px] font-normal font-sans leading-normal transition-colors", { "text-[#FDF277] font-semibold": day.isToday, "text-white": !day.isToday })}>
                {day.formattedDate}
            </p>

            {isOwner && (
                <div className="absolute top-[-2px] right-[10px] flex items-center gap-2 z-10">
                    <button
                        onClick={() => onAddSlot(day)}
                        className="flex items-center justify-center w-6 h-6 rounded-full bg-[#353333] hover:bg-green-600 transition-colors"
                        title="Добавить слот"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                             <line x1="12" y1="5" x2="12" y2="19"></line>
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                        </svg>
                    </button>
                     <button
                        onClick={() => onDeleteDaySlots(day)}
                        className="flex items-center justify-center w-6 h-6 rounded-full bg-[#353333] hover:bg-red-600 transition-colors"
                        title="Удалить все слоты дня"
                     >
                        <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                             <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
            )}

            {slotsCount === 0 ? (
                <p className="absolute top-[21px] left-[10px] text-[#595757] text-[10px] font-normal font-sans leading-none">
                    Нет запланированных слотов
                </p>
            ) : (
                <div className="absolute top-[33px] left-[10px] right-[10px] flex flex-col gap-[6px]">
                    {slots.map((slot) => (
                        <TimeSlotComponent
                            key={slot.id}
                            slot={slot}
                            day={day}
                            onTakeSlot={onTakeSlot}
                            onDeleteSlot={onDeleteSlot} // <--- ИЗМЕНЕНО НАЗВАНИЕ
                            currentUserId={currentUserId}
                            isOwner={isOwner}
                        />
                    ))}
                </div>
            )}
        </div>
    );
}

export default function Main({
    weekDays = [],
    onNextWeek,
    onPrevWeek,
    isLoading = false,
    currentUserId,
    onTakeSlot,
    onDeleteSlot, // <--- ИЗМЕНЕНО НАЗВАНИЕ
    onAddSlot,
    onDeleteDaySlots,
    isOwner,
}: MainProps) {

    const swipeHandlers: SwipeableHandlers = useSwipeable({
        onSwipedLeft: onNextWeek,
        onSwipedRight: onPrevWeek,
        preventScrollOnSwipe: true
    });

    if (isLoading) {
        return (<main className="container mx-auto bg-black rounded-lg w-full animate-pulse"><div className="flex flex-col pt-2">{[...Array(7)].map((_, i) => (<div key={i} className="h-[50px] relative border-b border-gray-800 last:border-b-0"><div className="absolute top-1 left-2 h-4 bg-gray-700 rounded w-1/3"></div></div>))}</div></main>);
    }

    return (
        <main {...swipeHandlers} className="container mx-auto bg-black rounded-lg w-full" style={{ height: 'calc(100vh - 180px)', overflowY: 'auto' }}>
            <div className="flex flex-col pt-2">
                {weekDays.length > 0 ? (
                    weekDays.map((day) => (
                        <DayRow
                            key={day.formattedDate}
                            day={day}
                            currentUserId={currentUserId}
                            onTakeSlot={onTakeSlot}
                            onDeleteSlot={onDeleteSlot} // <--- ИЗМЕНЕНО НАЗВАНИЕ
                            onAddSlot={onAddSlot}
                            onDeleteDaySlots={onDeleteDaySlots}
                            isOwner={isOwner}
                        />
                    ))
                ) : (
                     <div className="flex items-center justify-center h-full text-gray-500">Нет данных для отображения.</div>
                )}
            </div>
        </main>
    );
}