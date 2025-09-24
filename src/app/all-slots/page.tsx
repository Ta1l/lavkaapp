// src/app/all-slots/page.tsx

"use client";

import { useState } from "react";

export default function AllSlotsPage() {
    const [currentDay, setCurrentDay] = useState("Понедельник, 22 сентября");

    const handlePrevDay = () => {
        // Логика для переключения на предыдущий день
        console.log("Previous day");
    };

    const handleNextDay = () => {
        // Логика для переключения на следующий день
        console.log("Next day");
    };

    return (
        <div className="min-h-screen bg-black text-white">
            {/* Навигация по дням */}
            <div className="relative w-full pt-[43px]">
                {/* Кнопка влево */}
                <button
                    onClick={handlePrevDay}
                    className="absolute left-[14px] top-[43px] w-[25px] h-[25px] active:scale-95 transition-transform"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 25 25" fill="none">
                        <circle cx="12.5" cy="12.5" r="12.5" fill="#353333"/>
                        <path d="M12.2703 12L16 15.8333L14.8649 17L10 12L14.8649 7L16 8.16667L12.2703 12Z" fill="white"/>
                    </svg>
                </button>

                {/* Поле с днем недели */}
                <div className="mx-auto w-[246px] h-[27px] rounded-[20px] bg-[#353333] flex items-center justify-center">
                    <span className="text-white text-[14px] font-normal font-['Inter'] leading-normal">
                        {currentDay}
                    </span>
                </div>

                {/* Кнопка вправо */}
                <button
                    onClick={handleNextDay}
                    className="absolute right-[14px] top-[43px] w-[25px] h-[25px] active:scale-95 transition-transform"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 25 25" fill="none">
                        <circle cx="12.5" cy="12.5" r="12.5" fill="#353333"/>
                        <path d="M13.7297 12L10 8.16667L11.1351 7L16 12L11.1351 17L10 15.8333L13.7297 12Z" fill="white"/>
                    </svg>
                </button>
            </div>

            {/* Контент страницы */}
            <div className="container mx-auto p-4 mt-8">
                <div className="text-gray-400 text-center">
                    Здесь будут отображаться слоты всех пользователей
                </div>
            </div>
        </div>
    );
}