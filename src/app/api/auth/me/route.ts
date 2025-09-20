// src/app/api/auth/me/route.ts
import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";

export async function GET(req: NextRequest) {
  try {
    const auth = req.headers.get("authorization");
    if (!auth?.startsWith("Bearer ")) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const apiKey = auth.slice(7);

    const result = await pool.query(
      "SELECT id, username, full_name FROM users WHERE api_key = $1",
      [apiKey]
    );

    if (result.rowCount === 0) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const user = result.rows[0];
    return NextResponse.json(user, { status: 200 });
  } catch (err) {
    console.error("[API Auth Me Error]", err);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}