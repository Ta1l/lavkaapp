// src/components/DualTimePicker.tsx
"use client";

import React, { useState } from 'react';
import TimePicker from './TimePicker';

interface DualTimePickerProps {
  onStartTimeSelect: (time: string) => void;
  onEndTimeSelect: (time: string) => void;
  initialStartTime?: string;
  initialEndTime?: string;
}

export default function DualTimePicker({
  onStartTimeSelect,
  onEndTimeSelect,
  initialStartTime = "08:00",
  initialEndTime = "09:00"
}: DualTimePickerProps) {
  const [startTime, setStartTime] = useState(initialStartTime);
  const [endTime, setEndTime] = useState(initialEndTime);

  const handleStartTimeSelect = (time: string) => {
    setStartTime(time);
    onStartTimeSelect(time);
  };

  const handleEndTimeSelect = (time: string) => {
    setEndTime(time);
    onEndTimeSelect(time);
  };

  return (
    <div className="grid grid-cols-2 gap-5">
      <div className="w-full">
        <TimePicker
          onTimeSelect={handleStartTimeSelect}
          initialTime={startTime}
          label="Start time"
          dropdownPosition="left"
        />
      </div>
      <div className="w-full">
        <TimePicker
          onTimeSelect={handleEndTimeSelect}
          initialTime={endTime}
          label="End time"
          dropdownPosition="right"
        />
      </div>
    </div>
  );
} 