// src/lib/session.ts
import { cookies } from "next/headers";
import type { User } from "@/types/shifts";

export async function getUserFromSession(): Promise<User | null> {
  try {
    const cookie = cookies().get("auth-session");
    if (!cookie) return null;
    const parsed = JSON.parse(cookie.value);
    if (!parsed || !parsed.id) return null;
    // Возвращаем минимум полей: id, username, full_name
    return {
      id: parsed.id,
      username: parsed.username,
      full_name: parsed.full_name,
    } as User;
  } catch (err) {
    console.error("[getUserFromSession] error", err);
    return null;
  }
}
