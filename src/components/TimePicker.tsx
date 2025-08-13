// src/components/TimePicker.tsx

"use client";

import React, { useState, useEffect, useRef } from 'react';

interface TimePickerProps {
  onTimeSelect: (time: string) => void;
  initialTime?: string;
  label?: string;
  dropdownPosition?: 'left' | 'right';
}

export default function TimePicker({
  onTimeSelect,
  initialTime = "08:00",
  label = "Select time",
  dropdownPosition = 'left'
}: TimePickerProps) {
  const [selectedTime, setSelectedTime] = useState(initialTime);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const timeDisplayRef = useRef<HTMLButtonElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Generate time options from 08:00 to 23:00
  const timeOptions = Array.from({ length: 16 }, (_, i) => {
    const hour = i + 8;
    return `${hour.toString().padStart(2, '0')}:00`;
  });

  // Handle click outside to close dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        timeDisplayRef.current &&
        !timeDisplayRef.current.contains(event.target as Node)
      ) {
        setIsDropdownOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleTimeClick = (time: string) => {
    setSelectedTime(time);
    onTimeSelect(time);
    setIsDropdownOpen(false);
  };

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  return (
    <div className="relative w-full">
      <button
        ref={timeDisplayRef}
        onClick={toggleDropdown}
        className="w-full px-4 py-3 text-left text-white hover:text-[#FDF277] font-sans text-base transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#FDF277] focus-visible:ring-opacity-50"
        aria-haspopup="listbox"
        aria-expanded={isDropdownOpen}
        aria-label={label}
      >
        {selectedTime}
      </button>

      <div
        ref={dropdownRef}
        className={`absolute z-50 w-full bg-[#4B4747] rounded-[5px] overflow-y-auto transition-all duration-200 ease-in-out shadow-lg 
          [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]
          ${isDropdownOpen 
            ? 'opacity-100 translate-y-1 max-h-[150px]' 
            : 'opacity-0 -translate-y-2 max-h-0 pointer-events-none'
          }`}
        role="listbox"
        aria-label={`${label} options`}
      >
        {timeOptions.map((time) => (
          <button
            key={time}
            onClick={() => handleTimeClick(time)}
            className={`w-full px-4 py-3 text-left text-white hover:bg-[#595757] transition-colors duration-150 font-sans text-base
              ${selectedTime === time ? 'bg-[#595757]' : ''}
              focus-visible:outline-none focus-visible:bg-[#595757]`}
            role="option"
            aria-selected={selectedTime === time}
          >
            {time}
          </button>
        ))}
      </div>
    </div>
  );
}

// Add this to your globals.css file:
/*
@layer utilities {
  .scrollbar-hide {
    -ms-overflow-style: none;
    scrollbar-width: none;
  }
  .scrollbar-hide::-webkit-scrollbar {
    display: none;
  }
}
*/ 