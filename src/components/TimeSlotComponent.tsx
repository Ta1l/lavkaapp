// src/components/TimeSlotComponent.tsx

"use client";

import React from "react";
import { TimeSlot, Day } from "@/types/shifts";
import clsx from "clsx";

interface TimeSlotComponentProps {
    slot: TimeSlot;
    day: Day;
    onTakeSlot: (day: Day, slot: TimeSlot) => void;
    onDeleteSlot: (day: Day, slotId: number) => void;
    onEditSlot?: (day: Day, slot: TimeSlot) => void; // Новый проп
    currentUserId: number | null;
    isOwner: boolean;
}

export default function TimeSlotComponent({
    slot,
    day,
    onTakeSlot,
    onDeleteSlot,
    onEditSlot, // Новый проп
    currentUserId,
    isOwner,
}: TimeSlotComponentProps) {
    const isSlotAvailable = slot.user_id === null;
    const isSlotTaken = slot.user_id !== null;
    const isTakenByOther = slot.user_id !== null && slot.user_id !== currentUserId;
    const canBeTaken = isSlotAvailable && currentUserId !== null;
    
    // Владелец может редактировать любой слот
    const canEdit = isOwner && onEditSlot;

    const handleClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        
        // Если владелец - открываем редактирование
        if (canEdit) {
            onEditSlot(day, slot);
        } 
        // Если обычный пользователь и слот свободен - занимаем
        else if (canBeTaken) {
            onTakeSlot(day, slot);
        }
    };
    
    const canDelete = isSlotTaken && isOwner;

    return (
        <div
            className={clsx(
                "w-full h-[35px] rounded-[20px] flex items-center justify-between px-3 transition-all duration-200 relative",
                {
                    "bg-[#353333]": isSlotAvailable,
                    "cursor-pointer hover:bg-[#4a4848]": canBeTaken,
                    "bg-[#FFEA00]": isSlotTaken,
                    "cursor-pointer hover:bg-[#e6d400]": canEdit && isSlotTaken,
                    "cursor-default": !canBeTaken && !canEdit,
                }
            )}
            onClick={handleClick}
            title={
                canEdit ? "Нажмите для редактирования" :
                canBeTaken ? "Нажмите, чтобы занять слот" : 
                (slot.userName ? `Занято: ${slot.userName}` : "")
            }
        >
            <div className="flex items-center gap-2">
                <div className={clsx(
                    "text-center font-sans text-[14px] leading-normal",
                    {
                        "text-white": isSlotAvailable,
                        "text-black font-bold": isSlotTaken,
                    }
                )}>
                    {`${slot.startTime} - ${slot.endTime}`}
                </div>
                
                {/* Иконка редактирования для владельца */}
                {canEdit && (
                    <svg 
                        className={clsx(
                            "w-4 h-4",
                            {
                                "text-gray-400": isSlotAvailable,
                                "text-black/60": isSlotTaken,
                            }
                        )}
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                    >
                        <path 
                            strokeLinecap="round" 
                            strokeLinejoin="round" 
                            strokeWidth={2} 
                            d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" 
                        />
                    </svg>
                )}
            </div>
            
            {isTakenByOther && slot.userName && (
                <div className="text-black/70 text-xs font-light truncate max-w-[50%]">
                    {slot.userName}
                </div>
            )}
            
            {canDelete && (
                <button
                    type="button"
                    onClick={(e) => {
                        e.stopPropagation();
                        if (slot.id) {
                            onDeleteSlot(day, slot.id);
                        }
                    }}
                    className="p-1 rounded-full text-black/60 hover:text-white hover:bg-red-600 active:scale-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
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