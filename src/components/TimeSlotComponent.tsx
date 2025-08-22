// src/components/TimeSlotComponent.tsx

"use client";

import React from "react";
import { TimeSlot, Day } from "@/types/shifts";
import clsx from "clsx";

interface TimeSlotComponentProps {
    slot: TimeSlot;
    day: Day;
    onTakeSlot: (day: Day, slot: TimeSlot) => void;
    onDeleteSlot: (day: Day, slotId: number) => void; // <--- ИЗМЕНЕНО НАЗВАНИЕ
    currentUserId: number | null;
    isOwner: boolean;
}

export default function TimeSlotComponent({
    slot,
    day,
    onTakeSlot,
    onDeleteSlot, // <--- ИЗМЕНЕНО НАЗВАНИЕ
    currentUserId,
    isOwner,
}: TimeSlotComponentProps) {
    const isSlotAvailable = slot.user_id === null;
    const isSlotTaken = slot.user_id !== null;
    
    // --- НАЧАЛО ИЗМЕНЕНИЙ ---
    // Вычисляем, занят ли слот ДРУГИМ пользователем
    const isTakenByOther = slot.user_id !== null && slot.user_id !== currentUserId;
    // --- КОНЕЦ ИЗМЕНЕНИЙ ---

    const canBeTaken = isSlotAvailable && currentUserId !== null;

    const handleClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        if (canBeTaken) {
            onTakeSlot(day, slot);
        }
    };
    
    // Владелец расписания видит крестик на любом занятом слоте
    const canDelete = isSlotTaken && isOwner;

    return (
        <div
            className={clsx(
                "w-full h-[35px] rounded-[20px] flex items-center justify-between px-3 transition-all duration-200 relative",
                {
                    "bg-[#353333]": isSlotAvailable,
                    "cursor-pointer hover:bg-[#4a4848]": canBeTaken,
                    "bg-gray-700": isSlotTaken,
                    "cursor-default": !canBeTaken,
                }
            )}
            onClick={handleClick}
            title={canBeTaken ? "Нажмите, чтобы занять слот" : (slot.userName ? `Занято: ${slot.userName}` : "")}
        >
            <div className="text-white text-center font-sans text-[14px]">
                {`${slot.startTime} - ${slot.endTime}`}
            </div>
            
            {/* --- ИЗМЕНЕНИЕ: Показываем имя, только если слот занят ДРУГИМ пользователем --- */}
            {isTakenByOther && slot.userName && (
                <div className="text-gray-300 text-xs font-light truncate max-w-[50%]">
                    {slot.userName}
                </div>
            )}
            
            {/* Кнопка теперь вызывает onDeleteSlot */}
            {canDelete && (
                 <button
                    type="button"
                    onClick={(e) => {
                        e.stopPropagation();
                        if (slot.id) {
                            onDeleteSlot(day, slot.id); // <--- ИЗМЕНЕНО НАЗВАНИЕ
                        }
                    }}
                    className="p-1 rounded-full text-gray-400 hover:text-white hover:bg-red-600 active:scale-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
                    title="Удалить слот"
                >
                     <svg
                        xmlns="http://www.w3.org/2000/svg"
                        height="18px"
                        viewBox="0 -960 960 960"
                        width="18px"
                        fill="currentColor"
                    >
                        <path d="m256-200-56-56 224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z" />
                     </svg>
                </button>
            )}
        </div>
    );
}