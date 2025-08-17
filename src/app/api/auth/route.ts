// src/app/api/auth/route.ts
import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import { cookies } from "next/headers";
import bcrypt from "bcrypt";

const SALT_ROUNDS = 10;

// Обработчик формы авторизации/регистрации
export async function POST(request: NextRequest) {
  try {
    // ВАЖНО: читаем именно FormData, т.к. отправка идёт обычной HTML-формой
    const formData = await request.formData();
    const username = (formData.get("username") as string)?.trim();
    const password = (formData.get("password") as string) || "";
    const action = formData.get("action") as "login" | "register" | null;

    if (!username || !password || !action) {
      return NextResponse.redirect(new URL("/?error=validation_failed", request.url));
    }

    const existing = await pool.query("SELECT * FROM users WHERE username = $1", [username]);
    const existingUser = existing.rows[0] as { id: number; username: string; password: string } | undefined;

    let user: { id: number; username: string } | null = null;

    if (action === "register") {
      if (existingUser) {
        return NextResponse.redirect(new URL("/?error=user_exists", request.url));
      }
      const hash = await bcrypt.hash(password, SALT_ROUNDS);
      const inserted = await pool.query(
        "INSERT INTO users (username, password) VALUES ($1, $2) RETURNING id, username",
        [username, hash]
      );
      user = inserted.rows[0];
    } else if (action === "login") {
      if (!existingUser) {
        return NextResponse.redirect(new URL("/?error=invalid_credentials", request.url));
      }
      const ok = await bcrypt.compare(password, existingUser.password);
      if (!ok) {
        return NextResponse.redirect(new URL("/?error=invalid_credentials", request.url));
      }
      user = { id: existingUser.id, username: existingUser.username };
    } else {
      return NextResponse.redirect(new URL("/?error=invalid_action", request.url));
    }

    // Устанавливаем cookie-сессию и редиректим на расписание
    if (user) {
      const sessionData = { id: user.id, username: user.username };
      cookies().set("auth-session", JSON.stringify(sessionData), {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        maxAge: 60 * 60 * 24 * 7,
        path: "/",
      });
      return NextResponse.redirect(new URL("/schedule/0", request.url));
    }

    return NextResponse.redirect(new URL("/?error=unknown", request.url));
  } catch (err) {
    console.error("[API Auth Error]", err);
    return NextResponse.redirect(new URL("/?error=server_error", request.url));
  }
}
