// src/components/DayRow.tsx

"use client";

import React from "react";
import TimeSlotComponent from "./TimeSlotComponent";
import { Day, TimeSlot } from "@/types/shifts";
import { calculateDayHours } from "@/utils/calcDayHours";

interface DayRowProps {
    day: Day;
    currentUserId: number | null;
    onTakeSlot: (day: Day, slot: TimeSlot) => void;
    onDeleteSlot: (day: Day, slotId: number) => void;
    onEditSlot?: (day: Day, slot: TimeSlot) => void;
    onAddSlot: (day: Day) => void;
    onDeleteDaySlots: (day: Day) => void;
    isOwner: boolean;
}

export default function DayRow({
    day,
    currentUserId,
    onTakeSlot,
    onDeleteSlot,
    onEditSlot,
    onAddSlot,
    onDeleteDaySlots,
    isOwner,
}: DayRowProps) {
    const slots = day.slots || [];
    const slotsCount = slots.length;
    
    const calculateHeight = (count: number) => {
        if (count === 0) return 51;
        return 10 + 14 + 17 + (count * 41) - 6 + 10;
    };

    const mainHeight = calculateHeight(slotsCount);

    // Подготовка слотов для утилиты (поддерживаем разные названия полей)
    const simpleSlots = (slots || []).map((s: TimeSlot) => ({
        startTime: (s as any).startTime ?? (s as any).start_time ?? "",
        endTime: (s as any).endTime ?? (s as any).end_time ?? "",
    }));

    const totalHours = calculateDayHours(simpleSlots);

    const renderHoursText = (h: number) => {
        if (h === 0) return "";
        const isInt = Number.isInteger(h);
        return isInt ? `${h} ч` : `${h.toString().replace(/\.0+$/, "")} ч`;
    };

    // vertical center inside visible blue area (30px) with text height 12px -> top = 9px
    const blueVisibleTop = 30;
    const textHeight = 12;
    const textTop = Math.max(2, Math.round((blueVisibleTop - textHeight) / 2)); // fallback to 2px if needed

    return (
        <div className="relative w-full mt-[30px] mb-[10px] last:mb-0 overflow-visible">
            {/* Синяя карточка (за желтой), выглядывает на 30px */}
            <div 
                className="absolute w-full rounded-[20px] -top-[30px] left-0"
                style={{ 
                    backgroundColor: '#2C00C9E5',
                    height: `${mainHeight + 30}px`,
                    zIndex: 0
                }}
            >
                {/* Показываем часы только если есть > 0 */}
                {totalHours > 0 && (
                    <p
                        // позиционируем по правому краю (2px) и вертикально центрируем в видимой синей области
                        className="absolute font-sans font-bold leading-none text-black"
                        style={{
                            top: `${textTop}px`,       // центрируем в видимой синей части (или 2px)
                            right: "2px",              // отступ справа 2px
                            fontSize: "12px",
                            lineHeight: "12px",
                            zIndex: 2,
                        }}
                    >
                        {renderHoursText(totalHours)}
                    </p>
                )}
            </div>
            
            {/* Желтая карточка */}
            <div 
                className="relative w-full transition-all duration-300 rounded-[20px]"
                style={{ 
                    backgroundColor: '#E2CF00',
                    minHeight: `${mainHeight}px`,
                    zIndex: 1,
                }}
            >
                <p className="absolute top-[10px] left-[15px] text-[14px] font-bold font-sans leading-normal text-black">
                    {day.formattedDate}
                </p>

                {isOwner && (
                    <div className="absolute top-[7px] right-[15px] flex items-center gap-2 z-20">
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                onAddSlot(day);
                            }}
                            className="flex items-center justify-center w-7 h-7 rounded-full bg-black/20 hover:bg-green-600 transition-all duration-200 shadow-md"
                            title="Добавить слот"
                        >
                            <svg 
                                xmlns="http://www.w3.org/2000/svg" 
                                width="14" 
                                height="14" 
                                viewBox="0 0 24 24" 
                                fill="none" 
                                stroke="currentColor" 
                                strokeWidth="3" 
                                strokeLinecap="round" 
                                strokeLinejoin="round" 
                                className="text-white"
                            >
                                <line x1="12" y1="5" x2="12" y2="19"></line>
                                <line x1="5" y1="12" x2="19" y2="12"></line>
                            </svg>
                        </button>
                        
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                if (window.confirm('Удалить все слоты этого дня?')) {
                                    onDeleteDaySlots(day);
                                }
                            }}
                            className="flex items-center justify-center w-7 h-7 rounded-full bg-black/20 hover:bg-red-600 transition-all duration-200 shadow-md"
                            title="Удалить все слоты дня"
                        >
                            <svg 
                                xmlns="http://www.w3.org/2000/svg" 
                                width="12" 
                                height="12" 
                                viewBox="0 0 24 24" 
                                fill="none" 
                                stroke="currentColor" 
                                strokeWidth="3" 
                                strokeLinecap="round" 
                                strokeLinejoin="round" 
                                className="text-white"
                            >
                                <line x1="18" y1="6" x2="6" y2="18"></line>
                                <line x1="6" y1="6" x2="18" y2="18"></line>
                            </svg>
                        </button>
                    </div>
                )}

                {slotsCount === 0 ? (
                    <p className="absolute top-[31px] left-[15px] text-black/50 text-[10px] font-normal font-sans leading-none">
                        Нет запланированных слотов
                    </p>
                ) : (
                    <div className="absolute top-[41px] left-[15px] right-[15px] flex flex-col gap-[6px]">
                        {slots.map((slot) => (
                            <TimeSlotComponent
                                key={slot.id}
                                slot={slot}
                                day={day}
                                onTakeSlot={onTakeSlot}
                                onDeleteSlot={onDeleteSlot}
                                onEditSlot={onEditSlot}
                                currentUserId={currentUserId}
                                isOwner={isOwner}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
