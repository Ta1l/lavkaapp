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
    onEditSlot?: (day: Day, slot: TimeSlot) => void;
    currentUserId: number | null;
    isOwner: boolean;
}

export default function TimeSlotComponent({
    slot,
    day,
    onTakeSlot,
    onDeleteSlot,
    onEditSlot,
    currentUserId,
    isOwner,
}: TimeSlotComponentProps) {
    const isSlotAvailable = slot.user_id === null;
    const isSlotTaken = slot.user_id !== null;
    const isTakenByOther = slot.user_id !== null && slot.user_id !== currentUserId;
    const canBeTaken = isSlotAvailable && currentUserId !== null;
    
    const canEdit = isOwner && onEditSlot;

    const handleClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        
        if (canEdit) {
            onEditSlot(day, slot);
        } 
        else if (canBeTaken) {
            onTakeSlot(day, slot);
        }
    };
    
    const canDelete = isSlotTaken && isOwner;

    return (
        <div className="relative w-full">
            {/* Фоновая карточка синего цвета */}
            <div 
                className="absolute w-full h-[65px] rounded-[20px] bg-[#2C00C9E5] -top-[30px] left-0"
                style={{ zIndex: 0 }}
            />
            
            {/* Основная карточка */}
            <div
                className={clsx(
                    "relative w-full h-[35px] rounded-[20px] flex items-center justify-between px-3 transition-all duration-200",
                    {
                        "bg-[#E2CF00]": isSlotAvailable,
                        "cursor-pointer hover:bg-[#d4bf00]": canBeTaken,
                        "bg-[#000]": isSlotTaken,
                        "cursor-pointer hover:bg-[#333]": canEdit && isSlotTaken,
                        "cursor-default": !canBeTaken && !canEdit,
                    }
                )}
                style={{ zIndex: 1 }}
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
                            "text-black": isSlotAvailable,
                            "text-white font-bold": isSlotTaken,
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
                                    "text-black/60": isSlotAvailable,
                                    "text-white/60": isSlotTaken,
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
                    <div className="text-white/70 text-xs font-light truncate max-w-[50%]">
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
                        className="p-1 rounded-full text-white/60 hover:text-white hover:bg-red-600 active:scale-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
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
        </div>
    );
}