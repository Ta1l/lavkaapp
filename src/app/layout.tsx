// src/app/layout.tsx
import "./globals.css";
import type { Metadata } from "next";
import Script from 'next/script';

export const metadata: Metadata = {
  title: "Лавка",
  description: "Расписание смен",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ru" className="h-full">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Jura:wght@300;700&display=swap"
          rel="stylesheet"
        />
        {/* Добавляем Telegram WebApp SDK */}
        <Script
          src="https://telegram.org/js/telegram-web-app.js"
          strategy="beforeInteractive"
        />
      </head>
      <body className="min-h-full w-full antialiased bg-[var(--background)] text-[var(--foreground)]">
        <div className="flex min-h-full flex-col">{children}</div>
      </body>
    </html>
  );
}