// src/components/Main.tsx

"use client";

import { useSwipeable, SwipeableHandlers } from "react-swipeable";
import React from "react";
import DayRow from "./DayRow";
import { Day, TimeSlot } from "@/types/shifts";

interface MainProps {
    weekDays: Day[];
    onNextWeek: () => void;
    onPrevWeek: () => void;
    isLoading?: boolean;
    currentUserId: number | null;
    onTakeSlot: (day: Day, slot: TimeSlot) => void;
    onDeleteSlot: (day: Day, slotId: number) => void;
    onEditSlot?: (day: Day, slot: TimeSlot) => void;
    onAddSlot: (day: Day) => void;
    onDeleteDaySlots: (day: Day) => void;
    isOwner: boolean;
}

export default function Main({
    weekDays = [],
    onNextWeek,
    onPrevWeek,
    isLoading = false,
    currentUserId,
    onTakeSlot,
    onDeleteSlot,
    onEditSlot,
    onAddSlot,
    onDeleteDaySlots,
    isOwner,
}: MainProps) {
    console.log('🎨 Main component render:', {
        weekDaysCount: weekDays.length,
        totalSlots: weekDays.reduce((sum, day) => sum + (day.slots?.length || 0), 0),
        weekDays: weekDays.map(d => ({
            date: d.formattedDate,
            slotsCount: d.slots?.length || 0
        }))
    });

    console.log('Main component - isOwner:', isOwner, 'currentUserId:', currentUserId);

    const swipeHandlers: SwipeableHandlers = useSwipeable({
        onSwipedLeft: onNextWeek,
        onSwipedRight: onPrevWeek,
        preventScrollOnSwipe: true
    });

    if (isLoading) {
        return (
            <main className="container mx-auto bg-black rounded-lg w-full animate-pulse">
                <div className="flex flex-col pt-2 px-3">
                    {[...Array(7)].map((_, i) => (
                        <div key={i} className="relative rounded-[20px] bg-gray-700 mb-[15px] last:mb-0" style={{ minHeight: '50px' }}>
                            <div className="absolute top-[10px] left-[15px] h-4 bg-gray-600 rounded w-1/3"></div>
                        </div>
                    ))}
                </div>
            </main>
        );
    }

    return (
        <main 
            {...swipeHandlers} 
            className="container mx-auto bg-black rounded-lg w-full overflow-y-auto" 
            style={{ height: 'calc(100vh - 180px)' }}
        >
            <div className="flex flex-col pt-2 px-3 pb-3">
                {weekDays.length > 0 ? (
                    weekDays.map((day) => (
                        <DayRow
                            key={day.formattedDate}
                            day={day}
                            currentUserId={currentUserId}
                            onTakeSlot={onTakeSlot}
                            onDeleteSlot={onDeleteSlot}
                            onEditSlot={onEditSlot}
                            onAddSlot={onAddSlot}
                            onDeleteDaySlots={onDeleteDaySlots}
                            isOwner={isOwner}
                        />
                    ))
                ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                        Нет данных для отображения.
                    </div>
                )}
            </div>
        </main>
    );
}