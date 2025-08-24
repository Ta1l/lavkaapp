// src/app/api/auth/get-token/route.ts

import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import bcrypt from "bcrypt";
import crypto from "crypto";

/**
 * Обработчик GET-запросов.
 * Нужен для того, чтобы Next.js не редиректил автоматически.
 * Мы явно возвращаем 405 (Method Not Allowed).
 */
export async function GET() {
  return NextResponse.json(
    { error: "GET not allowed. Use POST instead." },
    { status: 405 }
  );
}

/**
 * Обработчик POST-запросов.
 * Аутентифицирует пользователя по логину/паролю и возвращает apiKey.
 */
export async function POST(request: NextRequest) {
  try {
    const { username, password } = await request.json();

    if (!username || !password) {
      return NextResponse.json(
        { error: "Username and password are required" },
        { status: 400 }
      );
    }

    // 1. Ищем пользователя в базе
    const result = await pool.query(
      "SELECT id, username, password, api_key FROM users WHERE username = $1",
      [username]
    );
    const user = result.rows[0];

    if (!user) {
      return NextResponse.json(
        { error: "Invalid credentials" },
        { status: 401 }
      );
    }

    // 2. Проверяем пароль
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return NextResponse.json(
        { error: "Invalid credentials" },
        { status: 401 }
      );
    }

    // 3. Проверяем/создаём API-ключ
    let apiKey = user.api_key;
    if (!apiKey) {
      apiKey = crypto.randomBytes(16).toString("hex");
      await pool.query("UPDATE users SET api_key = $1 WHERE id = $2", [
        apiKey,
        user.id,
      ]);
      console.log(`Generated new API key for user ${user.username}`);
    }

    // 4. Возвращаем ключ
    return NextResponse.json({ apiKey }, { status: 200 });

  } catch (err) {
    console.error("[API Get-Token Error]", err);
    return NextResponse.json(
      { error: "Server error" },
      { status: 500 }
    );
  }
}
