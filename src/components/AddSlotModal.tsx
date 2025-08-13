// src/components/AddSlotModal.tsx

"use client";

import React, { useEffect, useState } from "react";
import { useSwipeable, SwipeableHandlers } from "react-swipeable";
import DualTimePicker from "./DualTimePicker";

interface AddSlotModalProps {
  onClose: () => void;
  onDone: (startTime: string, endTime: string) => void;
  initialStartTime?: string;
  initialEndTime?: string;
}

export default function AddSlotModal({ 
  onClose, 
  onDone, 
  initialStartTime = "08:00",
  initialEndTime = "09:00" 
}: AddSlotModalProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [startTime, setStartTime] = useState<string>(initialStartTime);
  const [endTime, setEndTime] = useState<string>(initialEndTime);

  useEffect(() => {
    setIsMounted(true);
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  const swipeHandlers: SwipeableHandlers = useSwipeable({
    onSwipedDown: handleClose,
    preventScrollOnSwipe: true,
    trackMouse: false,
    delta: 10,
  });

  function handleClose() {
    setIsMounted(false);
  }

  function handleTransitionEnd() {
    if (!isMounted) {
      onClose();
    }
  }

  function handleStartTimeSelect(time: string) {
    setStartTime(time);
  }

  function handleEndTimeSelect(time: string) {
    setEndTime(time);
  }

  function handleDone() {
    onDone(startTime, endTime);
    handleClose();
  }

  return (
    <div
      {...swipeHandlers}
      onTransitionEnd={handleTransitionEnd}
      className={`fixed bottom-0 left-1/2 -translate-x-1/2 z-50 w-full h-[300px] rounded-[20px] bg-[#242424] transform transition-transform duration-300 ${
        isMounted ? "translate-y-0" : "translate-y-full"
      }`}
    >
      <p className="absolute top-[17px] left-[12px] w-[130px] h-[30px] text-white font-sans text-[20px] font-normal leading-none">
        Время слота
      </p>

      <div className="absolute top-[101px] left-[20px] w-[calc(100%-40px)]">
        <DualTimePicker
          onStartTimeSelect={handleStartTimeSelect}
          onEndTimeSelect={handleEndTimeSelect}
          initialStartTime={startTime}
          initialEndTime={endTime}
        />
      </div>

      <button
        type="button"
        onClick={handleClose}
        className="absolute left-[16px] top-[240px] w-[145px] h-[37px] bg-[#595757] rounded-[8px] active:scale-95 transition-transform duration-150 flex items-center justify-center"
      >
        <span className="w-[136px] h-[19px] text-white text-center font-sans text-[14px] font-normal leading-none">
          Закрыть
        </span>
      </button>

      <button
        type="button"
        onClick={handleDone}
        className="absolute right-[16px] top-[240px] w-[145px] h-[37px] bg-[#FDF277] rounded-[8px] active:scale-95 transition-transform duration-150 flex items-center justify-center"
      >
        <span className="w-[136px] h-[19px] text-black text-center font-sans text-[14px] font-normal leading-none">
          Готово
        </span>
      </button>
    </div>
  );
}