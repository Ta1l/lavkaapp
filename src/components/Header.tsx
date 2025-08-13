// components/Header.tsx

"use client";

import React from "react";

interface HeaderProps {
    dateRange: string;
    onNextWeek: () => void;
    onPrevWeek: () => void;
    isNextDisabled: boolean;
    isPrevDisabled: boolean;
    onLogout: () => void;
}

const Header: React.FC<HeaderProps> = ({
    dateRange,
    onNextWeek,
    onPrevWeek,
    isNextDisabled,
    isPrevDisabled,
    onLogout,
}) => {
    return (
        <header className="fixed top-0 left-0 w-full h-[100px] bg-black rounded-2xl z-10">
            <div className="absolute top-1/2 left-0 w-full h-[1px] bg-[#353333] -translate-y-1/2"></div>

            {/* [ИЗМЕНЕНО] Добавлен класс z-20, чтобы поднять кнопку на верхний слой */}
            <button
                type="button"
                onClick={onLogout}
                className="absolute top-[10px] right-[16px] w-[30px] h-[30px] flex items-center justify-center bg-[#353333] rounded-full active:scale-95 transition-transform duration-150 hover:bg-red-600 z-20"
                title="Выйти"
            >
                <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#FFF">
                    <path d="M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h280v80H200v560h280v80H200Zm440-160-55-58 102-102H360v-80h327L585-622l55-58 200 200-200 200Z"/>
                </svg>
            </button>

            <div className="absolute text-white top-[11px] left-1/2 -translate-x-1/2 font-sans text-[14px] font-normal">
                12 слотов · 57 часов
            </div>
            <div className="absolute text-white top-[27px] left-1/2 -translate-x-1/2 text-[10px] font-sans font-normal">
                В сумме за неделю
            </div>
            <div className="absolute inset-0">
                <button
                    type="button"
                    onClick={onPrevWeek}
                    disabled={isPrevDisabled}
                    className="absolute top-[66%] left-[4.73%] w-[25px] h-[25px] bg-[#353333] rounded-full flex items-center justify-center active:scale-95 transition-transform duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="6" height="10" viewBox="0 0 6 10" fill="none">
                        <path d="M2.27027 5L6 8.83333L4.86486 10L0 5L4.86487 0L6 1.16667L2.27027 5Z" fill="white" />
                    </svg>
                </button>
                <button
                    type="button"
                    onClick={onNextWeek}
                    disabled={isNextDisabled}
                    className="absolute top-[66%] left-[88.579%] w-[25px] h-[25px] bg-[#353333] rounded-full flex items-center justify-center active:scale-95 transition-transform duration-150 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="6" height="10" viewBox="0 0 6 10" fill="none">
                        <path d="M3.72973 5L0 1.16667L1.13514 0L6 5L1.13514 10L0 8.83333L3.72973 5Z" fill="white" />
                    </svg>
                </button>
            </div>
            <div className="absolute -translate-x-1/2 top-[66px] left-1/2 bg-[#353333] w-auto px-4 h-[27px] rounded-[20px] flex items-center justify-center">
                <span className="text-white text-center font-sans text-[10px] font-normal leading-none">
                    {dateRange}
                </span>
            </div>
        </header>
    );
};

export default Header;