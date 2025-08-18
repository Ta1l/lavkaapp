// src/app/api/slots/route.ts
import { NextRequest, NextResponse } from "next/server";
import { pool } from "@/lib/db";
import { getUserFromSession } from "@/lib/session";

/**
 * POST - бронирование слота (тело: { slotId: number })
 * DELETE - освобождение слота (query: ?id=...)
 */

export async function POST(request: NextRequest) {
  const user = await getUserFromSession();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const { slotId } = await request.json();
    if (!slotId) return NextResponse.json({ error: "Missing slotId" }, { status: 400 });

    // Ставим владельцем слота и помечаем как booked, только если он свободен
    const result = await pool.query(
      `UPDATE shifts
       SET user_id = $1, status = 'booked'
       WHERE id = $2 AND (user_id IS NULL OR user_id = $1)
       RETURNING id, user_id, shift_date, shift_code, status`,
      [user.id, slotId]
    );

    if ((result.rowCount ?? 0) === 0) {
      return NextResponse.json({ error: "Slot not available" }, { status: 409 });
    }

    // Возвращаем обновлённый слот
    return NextResponse.json(result.rows[0]);
  } catch (err) {
    console.error("[API SLOTS POST Error]", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  const user = await getUserFromSession();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const { searchParams } = new URL(request.url);
    const slotIdRaw = searchParams.get("id");
    if (!slotIdRaw) return NextResponse.json({ error: "Missing id param" }, { status: 400 });
    const slotId = parseInt(slotIdRaw, 10);

    const result = await pool.query(
      `UPDATE shifts
       SET user_id = NULL, status = 'available'
       WHERE id = $1 AND user_id = $2
       RETURNING id, user_id, shift_date, shift_code, status`,
      [slotId, user.id]
    );

    if ((result.rowCount ?? 0) === 0) {
      return NextResponse.json({ error: "Slot not found or not owned by you" }, { status: 404 });
    }

    return NextResponse.json({ ok: true, slot: result.rows[0] });
  } catch (err) {
    console.error("[API SLOTS DELETE Error]", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
