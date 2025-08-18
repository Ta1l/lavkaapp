// src/app/api/shifts/route.ts
import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import { format } from "date-fns";
import { getUserFromSession } from "@/lib/session";

/**
 * normalizeDateString - возвращает YYYY-MM-DD или null
 */
function normalizeDateString(d: string | Date): string | null {
  try {
    const dt = typeof d === "string" ? new Date(d) : d;
    if (Number.isNaN(dt.getTime())) return null;
    return format(dt, "yyyy-MM-dd");
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const viewedUserIdParam = url.searchParams.get("userId");
    const startParam = url.searchParams.get("start");
    const endParam = url.searchParams.get("end");

    const startStr = startParam ? normalizeDateString(startParam) : null;
    const endStr = endParam ? normalizeDateString(endParam) : null;

    const currentUser = await getUserFromSession();

    let query = `
      SELECT s.id, s.user_id, s.shift_date, s.shift_code, s.status,
             u.username
      FROM shifts s
      LEFT JOIN users u ON u.id = s.user_id
    `;
    const where: string[] = [];
    const params: any[] = [];

    if (startStr) {
      params.push(startStr);
      where.push(`s.shift_date >= $${params.length}`);
    }
    if (endStr) {
      params.push(endStr);
      where.push(`s.shift_date < $${params.length}`);
    }

    const isOwnerView =
      !viewedUserIdParam ||
      (currentUser && currentUser.id.toString() === viewedUserIdParam);

    if (isOwnerView && currentUser) {
      params.push(currentUser.id);
      where.push(`(s.user_id = $${params.length} OR s.status = 'available')`);
    } else if (viewedUserIdParam) {
      params.push(parseInt(viewedUserIdParam, 10));
      where.push(`s.user_id = $${params.length}`);
    } else {
      where.push(`s.status = 'available'`);
    }

    if (where.length) {
      query += " WHERE " + where.join(" AND ");
    }
    query += " ORDER BY s.shift_date, s.shift_code";

    const result = await pool.query(query, params);
    return NextResponse.json(result.rows);
  } catch (err) {
    console.error("[GET /api/shifts] error", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await request.json();
    const { date, startTime, endTime } = body ?? {};
    if (!date || !startTime || !endTime) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }

    const dateStr = normalizeDateString(date);
    if (!dateStr) {
      return NextResponse.json({ error: "Invalid date" }, { status: 400 });
    }

    const shiftCode = `${startTime}-${endTime}`;

    // Проверяем, есть ли такой тип слота
    const existing = await pool.query(
      "SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2",
      [dateStr, shiftCode]
    );

    if ((existing.rowCount ?? 0) > 0) {
      // Возвращаем существующий тип слота
      return NextResponse.json(existing.rows[0]);
    }

    // Создаём новый тип слота (user_id = NULL, status = 'available')
    const insert = await pool.query(
      `INSERT INTO shifts (shift_date, day_of_week, shift_code, status)
       VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, 'available')
       RETURNING *`,
      [dateStr, shiftCode]
    );
    return NextResponse.json(insert.rows[0], { status: 201 });
  } catch (err) {
    console.error("[POST /api/shifts] error", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const currentUser = await getUserFromSession();
    if (!currentUser) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const url = new URL(request.url);
    const dateParam = url.searchParams.get("date");
    if (!dateParam) {
      return NextResponse.json({ error: "Missing date param" }, { status: 400 });
    }

    const dateStr = normalizeDateString(dateParam);
    if (!dateStr) {
      return NextResponse.json({ error: "Invalid date" }, { status: 400 });
    }

    // Освобождаем все слоты пользователя за день
    const res = await pool.query(
      `UPDATE shifts SET user_id = NULL, status = 'available'
       WHERE shift_date = $1 AND user_id = $2`,
      [dateStr, currentUser.id]
    );

    return NextResponse.json({ ok: true, releasedCount: res.rowCount ?? 0 });
  } catch (err) {
    console.error("[DELETE /api/shifts] error", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
