// src/components/TopPageFooter.tsx

import React from "react";
import Link from "next/link";

const TopPageFooter: React.FC = () => {
  return (
    <footer className="bg-[#272727] rounded-[20px] w-full h-[54px] fixed bottom-0 left-0 flex items-center justify-center">
      <Link 
        href="/schedule/0"
        className="absolute left-[17px] top-[9px] w-[145px] h-[37px] bg-[#000] rounded-[8px] text-white text-[14px] font-sans font-normal leading-none active:scale-95 transition-transform duration-80 flex items-center justify-center"
      >
        Назад
      </Link>
      {/* Правая кнопка отсутствует, согласно ТЗ */}
    </footer>
  );
};

export default TopPageFooter;