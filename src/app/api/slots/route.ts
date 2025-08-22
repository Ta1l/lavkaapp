// src/app/api/slots/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { cookies } from 'next/headers';
import { User, Shift } from '@/types/shifts';

async function getUserFromSession(): Promise<User | null> {
    const sessionCookie = cookies().get('auth-session');
    if (!sessionCookie) return null;
    try {
        const sessionData = JSON.parse(sessionCookie.value);
        return sessionData.id ? sessionData : null;
    } catch { return null; }
}

export async function POST(request: NextRequest) {
    const user = await getUserFromSession();
    if (!user) return NextResponse.json({ error: 'Необходима авторизация' }, { status: 401 });
    try {
        const { slotId } = await request.json();
        if (!slotId) return NextResponse.json({ error: 'Не указан ID слота' }, { status: 400 });
        const result = await pool.query<Shift>(
            `UPDATE shifts SET user_id = $1, status = 'pending' WHERE id = $2 AND user_id IS NULL RETURNING *`,
            [user.id, slotId]
        );
        if (result.rowCount === 0) return NextResponse.json({ error: 'Слот уже занят или не существует' }, { status: 409 });
        return NextResponse.json(result.rows[0]);
    } catch (error) {
        console.error('[API SLOTS POST Error]', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}

// --- НАЧАЛО ИЗМЕНЕНИЙ ---
export async function DELETE(request: NextRequest) {
    const user = await getUserFromSession();
    if (!user) {
        return NextResponse.json({ error: 'Необходима авторизация' }, { status: 401 });
    }
    
    try {
        const { searchParams } = new URL(request.url);
        const slotId = searchParams.get('id');

        if (!slotId || isNaN(parseInt(slotId, 10))) {
            return NextResponse.json({ error: 'Не указан корректный ID слота' }, { status: 400 });
        }
        
        console.log(`[API SLOTS DELETE] User ${user.id} attempting to DELETE slot ${slotId}`);

        // ИЗМЕНЕНИЕ: Вместо UPDATE теперь DELETE. Слот удаляется навсегда.
        const result = await pool.query(
          `DELETE FROM shifts WHERE id = $1`,
          [parseInt(slotId, 10)]
        );

        if (result.rowCount === 0) {
            console.warn(`[API SLOTS DELETE] Warning: Slot with ID ${slotId} not found for deletion.`);
            return NextResponse.json({ error: 'Слот не найден' }, { status: 404 });
        }
        
        console.log(`[API SLOTS DELETE] Success: User ${user.id} DELETED slot ${slotId}`);
        return NextResponse.json({ message: 'Слот успешно удален' });

    } catch (error) {
        console.error('[API SLOTS DELETE Error]', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}
// --- КОНЕЦ ИЗМЕНЕНИЙ ---