// src/app/top/page.tsx
import React from "react";
import Link from "next/link";
import TopPageFooter from "@/components/TopPageFooter";
import { pool } from "@/lib/db";
import { User } from "@/types/shifts";
import { getUserFromSession } from "@/lib/session";

async function getUsers(): Promise<User[]> {
  try {
    const { rows } = await pool.query("SELECT id, username, full_name FROM users ORDER BY id");
    return rows;
  } catch (error) {
    console.error("Failed to fetch users:", error);
    return [];
  }
}

export default async function TopUsersPage() {
  const allUsers = await getUsers();
  const currentUser = await getUserFromSession();

  // Показываем всех кроме текущего пользователя (если он есть)
  const usersToShow = currentUser ? allUsers.filter((u) => u.id !== currentUser.id) : allUsers;

  return (
    <main className="bg-black min-h-screen rounded-[20px] text-white relative pb-[70px]">
      <div className="p-[25px] pt-[30px]">
        <h1 className="w-full text-[20px] font-normal font-sans mb-8">
          Список пользователей
        </h1>

        <div className="flex flex-col gap-4">
          {usersToShow.length > 0 ? (
            usersToShow.map((user, index) => (
              <Link href={`/schedule/0?userId=${user.id}`} key={user.id}>
                <div className="w-full bg-[#1C1C1C] rounded-lg p-4 flex items-center justify-between transition-colors hover:bg-[#2a2a2a]">
                  <div className="flex items-center gap-4">
                    <span className="text-gray-400 text-lg font-semibold">{index + 1}</span>
                    <span className="text-white text-base">{user.full_name || user.username}</span>
                  </div>
                  <span className="text-gray-500 text-sm">Посмотреть расписание →</span>
                </div>
              </Link>
            ))
          ) : (
            <p className="text-gray-500">Других пользователей пока нет.</p>
          )}
        </div>
      </div>
      <TopPageFooter />
    </main>
  );
}
