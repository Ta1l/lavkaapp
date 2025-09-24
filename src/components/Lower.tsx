// src/components/Lower.tsx

import React from "react";
import Link from "next/link";

interface LowerProps {
    onAddSlotClick: () => void;
    isOwner: boolean;
}

const Lower: React.FC<LowerProps> = ({ onAddSlotClick, isOwner }) => {
    return (
        <footer className="bg-[#272727] rounded-[20px] w-full h-[54px] fixed bottom-0 left-0 flex items-center justify-center">
            
            {isOwner ? (
                // Если пользователь на своей странице, показываем кнопку "Посмотреть топ"
                <Link 
                    href="/top" 
                    className="absolute left-[17px] top-[9px] w-[145px] h-[37px] bg-[#000] rounded-[8px] text-white text-[14px] font-sans font-normal leading-none active:scale-95 transition-transform duration-80 flex items-center justify-center"
                >
                    Посмотреть топ
                </Link>
            ) : (
                // Если пользователь на чужой странице, показываем кнопку "В профиль"
                <Link 
                    href="/schedule/0" 
                    className="absolute left-[17px] top-[9px] w-[145px] h-[37px] bg-[#000] rounded-[8px] text-white text-[14px] font-sans font-normal leading-none active:scale-95 transition-transform duration-80 flex items-center justify-center"
                >
                    В профиль
                </Link>
            )}

            {/* Кнопка "Все слоты" - показывается только для владельца расписания */}
            {isOwner && (
                <Link
                    href="/all-slots"
                    className="absolute right-[17px] top-[9px] w-[145px] h-[37px] bg-[#ffed23] rounded-[8px] text-black text-[14px] font-sans font-normal leading-none active:scale-95 transition-transform duration-80 flex items-center justify-center"
                >
                    Все слоты
                </Link>
            )}
        </footer>
    );
};

export default Lower;