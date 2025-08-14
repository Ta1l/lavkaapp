// src/components/TimeSlotComponent.tsx

"use client";

import React from "react";
import { TimeSlot, Day } from "@/types/shifts";
import clsx from "clsx";

interface TimeSlotComponentProps {
    slot: TimeSlot;
    day: Day;
    onTakeSlot: (day: Day, slot: TimeSlot) => void;
    onReleaseSlot: (day: Day, slotId: number) => void;
    currentUserId: number | null;
    isOwner: boolean;
}

export default function TimeSlotComponent({
    slot,
    day,
    onTakeSlot,
    onReleaseSlot,
    currentUserId,
    isOwner,
}: TimeSlotComponentProps) {

    // Определяем состояния слота
    const isSlotAvailable = slot.user_id === null;
    const isMySlot = slot.user_id === currentUserId && currentUserId !== null;
    const isTakenByOther = slot.user_id !== null && !isMySlot;

    // [ИСПРАВЛЕНО] Ключевое условие: можно ли занять этот слот?
    // Только если он свободен И мы являемся владельцем этого расписания.
    const canBeTaken = isSlotAvailable && isOwner;

    const handleClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        // Клик сработает только если слот можно занять
        if (canBeTaken) {
            onTakeSlot(day, slot);
        }
    };

    return (
        <div
            className={clsx(
                "w-full h-[35px] rounded-[20px] flex items-center justify-between px-3 transition-colors relative",
                {
                    "bg-[#353333]": isSlotAvailable,
                    "cursor-pointer hover:bg-[#404040]": canBeTaken, // Интерактивность только если можно занять
                    "cursor-default": !canBeTaken && isSlotAvailable, // Статичный курсор для свободных слотов на чужой странице
                    "bg-blue-800": isMySlot, // Мои слоты
                    "bg-gray-700 opacity-70": isTakenByOther, // Слоты, занятые другими
                }
            )}
            onClick={handleClick}
            title={canBeTaken ? "Нажмите, чтобы занять слот" : ""}
        >
            <div className="text-white text-center font-sans text-[14px]">
                {`${slot.startTime} - ${slot.endTime}`}
            </div>

            {isTakenByOther && slot.userName && (
                <div className="text-gray-300 text-xs font-light">
                    {slot.userName}
                </div>
            )}
            
            {/* Надпись "Свободен" показывается на свободных слотах на ЧУЖИХ страницах */}
            {isSlotAvailable && !isOwner && (
                 <div className="text-gray-300 text-xs font-light">
                    Свободен
                </div>
            )}

            {/* Кнопка удаления появляется только на МОИХ слотах И только в МОЕМ расписании */}
            {isMySlot && isOwner && (
                <button
                    type="button"
                    onClick={(e) => {
                        e.stopPropagation();
                        if (slot.id) {
                            onReleaseSlot(day, slot.id);
                        }
                    }}
                    className="p-1 rounded-full text-gray-400 hover:text-white hover:bg-red-600 active:scale-90"
                    title="Освободить слот"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="currentColor">
                        <path d="m256-200-56-56 224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z"/>
                    </svg>
                </button>
            )}
        </div>
    );
}