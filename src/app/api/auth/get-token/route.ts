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
    // Логируем входящий запрос
    console.log("[API Get-Token] Received POST request");
    
    const body = await request.json();
    console.log("[API Get-Token] Request body:", { username: body.username, password: "***" });
    
    const { username, password } = body;

    if (!username || !password) {
      console.log("[API Get-Token] Missing credentials");
      return NextResponse.json(
        { error: "Username and password are required" },
        { status: 400 }
      );
    }

    // 1. Ищем пользователя в базе
    console.log(`[API Get-Token] Looking for user: ${username}`);
    const result = await pool.query(
      "SELECT id, username, password, api_key FROM users WHERE username = $1",
      [username]
    );
    const user = result.rows[0];

    if (!user) {
      console.log(`[API Get-Token] User not found: ${username}`);
      return NextResponse.json(
        { error: "Invalid credentials" },
        { status: 401 }
      );
    }

    console.log(`[API Get-Token] User found: ${user.username} (id: ${user.id})`);

    // 2. Проверяем пароль
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      console.log(`[API Get-Token] Invalid password for user: ${username}`);
      return NextResponse.json(
        { error: "Invalid credentials" },
        { status: 401 }
      );
    }

    console.log(`[API Get-Token] Password valid for user: ${username}`);

    // 3. Проверяем/создаём API-ключ
    let apiKey = user.api_key;
    if (!apiKey) {
      // Генерируем новый API-ключ
      apiKey = crypto.randomBytes(32).toString("hex");
      await pool.query("UPDATE users SET api_key = $1 WHERE id = $2", [
        apiKey,
        user.id,
      ]);
      console.log(`[API Get-Token] Generated new API key for user ${user.username}`);
    } else {
      console.log(`[API Get-Token] Using existing API key for user ${user.username}`);
    }

    // 4. Возвращаем ключ
    console.log(`[API Get-Token] Returning API key for user ${user.username}`);
    return NextResponse.json(
      { 
        apiKey,
        message: "Authentication successful"
      }, 
      { status: 200 }
    );

  } catch (err) {
    console.error("[API Get-Token Error]", err);
    
    // Более детальная информация об ошибке для отладки
    if (err instanceof Error) {
      console.error("[API Get-Token Error Details]", {
        message: err.message,
        stack: err.stack,
        name: err.name
      });
    }
    
    return NextResponse.json(
      { error: "Server error" },
      { status: 500 }
    );
  }
}

/**
 * Обработчик OPTIONS-запросов для CORS
 */
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}