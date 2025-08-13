// src/app/top/layout.tsx

import React from "react";

export default function TopPageLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Простой layout, который просто рендерит дочернюю страницу без лишних элементов
  return <>{children}</>;
}