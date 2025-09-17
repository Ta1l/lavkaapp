// src/app/api/auth/get-token/route.ts

import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import bcrypt from "bcrypt";
import crypto from "crypto";

/**
 * GET — запрещён
 */
export async function GET() {
  return NextResponse.json(
    { error: "GET not allowed. Use POST instead." },
    { status: 405 }
  );
}

/**
 * POST — логин
 */
export async function POST(request: NextRequest) {
  try {
    console.log("[API Get-Token] Received POST request");

    const body = await request.json();
    const { username, password } = body;

    // Телеграм ID передаём через заголовок
    const telegramIdHeader = request.headers.get("x-telegram-id");
    console.log("[API Get-Token] Telegram ID header:", telegramIdHeader);

    if (!username || !password) {
      return NextResponse.json(
        { error: "Username and password are required" },
        { status: 400 }
      );
    }

    // 1. Ищем пользователя
    const result = await pool.query(
      "SELECT id, username, password, api_key, telegram_id FROM users WHERE username = $1",
      [username]
    );
    const user = result.rows[0];

    if (!user) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
    }

    // 2. Проверяем пароль
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
    }

    // 3. Генерируем apiKey, если его нет
    let apiKey = user.api_key;
    if (!apiKey) {
      apiKey = crypto.randomBytes(32).toString("hex");
      await pool.query("UPDATE users SET api_key = $1 WHERE id = $2", [
        apiKey,
        user.id,
      ]);
      console.log(`[API Get-Token] Generated new API key for ${user.username}`);
    }

    // 4. Если пришёл Telegram ID — привязываем, но только если у юзера ещё нет
    if (telegramIdHeader && !user.telegram_id) {
      const telegramId = BigInt(telegramIdHeader);
      await pool.query("UPDATE users SET telegram_id = $1 WHERE id = $2", [
        telegramId,
        user.id,
      ]);
      console.log(
        `[API Get-Token] Linked Telegram ID ${telegramId} with user ${user.username}`
      );
    }

    return NextResponse.json({ apiKey }, { status: 200 });
  } catch (err) {
    console.error("[API Get-Token Error]", err);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}

/**
 * OPTIONS для CORS
 */
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers":
        "Content-Type, x-telegram-id",
    },
  });
}
