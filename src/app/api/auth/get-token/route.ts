// src/app/api/auth/get-token/route.ts

import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import bcrypt from "bcrypt";
import crypto from "crypto";

export async function POST(request: NextRequest) {
  try {
    const { username, password } = await request.json();

    if (!username || !password) {
      return NextResponse.json({ error: "Username and password are required" }, { status: 400 });
    }

    // 1. Ищем пользователя в базе
    const result = await pool.query("SELECT id, username, password, api_key FROM users WHERE username = $1", [username]);
    const user = result.rows[0];

    if (!user) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
    }

    // 2. Проверяем пароль
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
    }

    let apiKey = user.api_key;

    // 3. Если у пользователя еще нет API-ключа, генерируем и сохраняем его
    if (!apiKey) {
      apiKey = crypto.randomBytes(16).toString('hex');
      await pool.query("UPDATE users SET api_key = $1 WHERE id = $2", [apiKey, user.id]);
      console.log(`Generated new API key for user ${user.username}`);
    }

    // 4. Возвращаем ключ боту
    return NextResponse.json({ apiKey: apiKey });

  } catch (err) {
    console.error("[API Get-Token Error]", err);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}