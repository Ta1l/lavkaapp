// src/app/api/auth/auto-login/route.ts

import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

export async function POST(request: NextRequest) {
  try {
    const { telegramId } = await request.json();

    if (!telegramId) {
      return NextResponse.json({ error: "telegramId required" }, { status: 400 });
    }

    const result = await pool.query(
      "SELECT api_key FROM users WHERE telegram_id = $1",
      [telegramId]
    );

    if (result.rowCount === 0) {
      return NextResponse.json({ error: "Not linked" }, { status: 404 });
    }

    return NextResponse.json({ apiKey: result.rows[0].api_key }, { status: 200 });
  } catch (err) {
    console.error("[API Auto-Login Error]", err);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}
