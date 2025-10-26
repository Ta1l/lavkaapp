// src/components/DayRow.tsx

"use client";

import React, { useEffect, useState } from "react";
import TimeSlotComponent from "./TimeSlotComponent";
import { Day, TimeSlot } from "@/types/shifts";
import { calculateDayHours } from "@/utils/calcDayHours";
import { subscribe as subscribeWeather, getWeatherForDate } from "@/utils/weather";

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

/**
 * Универсальная функция: получает day (объект Day из types/shifts)
 * и пытается вернуть ISO дату "YYYY-MM-DD".
 * Поддерживаются:
 *  - day.date как Date объект
 *  - day.date как timestamp number
 *  - day.date как строка "YYYY-MM-DD..."
 *  - поля isoDate / iso / time / startDate / formattedDate
 */
function getIsoDateFromDay(day: any): string | null {
    if (!day) return null;

    // 1) date как Date
    if (day.date instanceof Date) {
        return day.date.toISOString().slice(0, 10);
    }

    // 2) date как timestamp number
    if (typeof day.date === "number") {
        return new Date(day.date).toISOString().slice(0, 10);
    }

    // 3) date как строка 'YYYY-MM-DD...' (например ISO)
    if (typeof day.date === "string" && /^\d{4}-\d{2}-\d{2}/.test(day.date)) {
        return day.date.substring(0, 10);
    }

    // 4) другие поля строки
    const candidates = ["isoDate", "iso", "time", "startDate", "formattedDate", "dateStr"];
    for (const key of candidates) {
        const val = day[key];
        if (typeof val === "string" && /^\d{4}-\d{2}-\d{2}/.test(val)) {
            return val.substring(0, 10);
        }
    }

    return null;
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

    // Синия видимая область = 30px; центрируем вертикально
    const blueVisibleHeight = 30;
    const labelFontSize = 13;
    const labelTop = Math.max(2, Math.round((blueVisibleHeight - labelFontSize) / 2));

    // Получаем ISO дату карточки (YYYY-MM-DD)
    const isoDate = getIsoDateFromDay(day);

    // Состояние погоды для этой карточки
    const [weatherText, setWeatherText] = useState<string | null>(() => {
        if (!isoDate) return null;
        return getWeatherForDate(isoDate);
    });

    useEffect(() => {
        // подпишемся на глобальные обновления прогноза (получаем полный dailyMap)
        const unsub = subscribeWeather((dailyMap: Record<string, string>) => {
            if (!isoDate) return;
            const val = dailyMap ? dailyMap[isoDate] ?? null : null;
            setWeatherText(val);
        });
        return () => {
            try { unsub(); } catch (e) { /* ignore */ }
        };
    }, [isoDate]);

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
                {/* Погода — по центру видимой синей области, отступ слева 2px */}
                {weatherText && (
                    <p
                        className="absolute font-sans font-bold leading-none text-black"
                        style={{
                            top: `${labelTop}px`,
                            left: "2px",
                            fontSize: `${labelFontSize}px`,
                            lineHeight: `${labelFontSize}px`,
                            zIndex: 2,
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            maxWidth: "60%",
                            textAlign: "left",
                        }}
                        title={weatherText}
                    >
                        {weatherText}
                    </p>
                )}

                {/* Результат calcDayHours — по центру видимой синей области, отступ справа 2px */}
                {totalHours > 0 && (
                    <p
                        className="absolute font-sans font-bold leading-none text-black"
                        style={{
                            top: `${labelTop}px`,
                            right: "2px",
                            fontSize: `${labelFontSize}px`,
                            lineHeight: `${labelFontSize}px`,
                            zIndex: 2,
                            whiteSpace: "nowrap",
                        }}
                        title={renderHoursText(totalHours)}
                    >
                        {renderHoursText(totalHours)}
                    </p>
                )}
            </div>
            
            {/* Желтая карточка */}
            <div 
                className="relative w-full transition-all duration-300 rounded-[20px]"
                style={{ 
                    backgroundColor: '#E2CF00E5',
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
