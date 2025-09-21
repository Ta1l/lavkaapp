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

async function getUserFromRequest(request: NextRequest): Promise<User | null> {
    // Сначала проверяем API ключ (для бота)
    const authHeader = request.headers.get('Authorization');
    if (authHeader && authHeader.startsWith('Bearer ')) {
        const apiKey = authHeader.substring(7);
        if (apiKey) {
            const { rows } = await pool.query<User>('SELECT id, username, full_name FROM users WHERE api_key = $1', [apiKey]);
            if (rows.length > 0) {
                return rows[0];
            }
        }
    }
    
    // Затем проверяем сессию (для веб-приложения)
    return getUserFromSession();
}

export async function POST(request: NextRequest) {
    const user = await getUserFromRequest(request);
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

export async function PATCH(request: NextRequest) {
    const user = await getUserFromRequest(request);
    if (!user) {
        return NextResponse.json({ error: 'Необходима авторизация' }, { status: 401 });
    }

    try {
        const { slotId, startTime, endTime } = await request.json();
        
        if (!slotId || !startTime || !endTime) {
            return NextResponse.json({ error: 'Необходимо указать slotId, startTime и endTime' }, { status: 400 });
        }

        // Проверяем, что слот принадлежит пользователю
        const checkResult = await pool.query(
            'SELECT user_id FROM shifts WHERE id = $1',
            [slotId]
        );
        
        if (checkResult.rowCount === 0) {
            return NextResponse.json({ error: 'Слот не найден' }, { status: 404 });
        }
        
        const slot = checkResult.rows[0];
        
        // Пользователь может редактировать только свои слоты
        if (slot.user_id !== user.id) {
            return NextResponse.json({ error: 'Вы можете редактировать только свои слоты' }, { status: 403 });
        }

        const shiftCode = `${startTime}-${endTime}`;
        
        console.log(`[API SLOTS PATCH] User ${user.id} updating slot ${slotId} to ${shiftCode}`);

        // Обновляем слот
        const result = await pool.query<Shift>(
            'UPDATE shifts SET shift_code = $1, updated_at = NOW() WHERE id = $2 AND user_id = $3 RETURNING *',
            [shiftCode, slotId, user.id]
        );

        if (result.rowCount === 0) {
            return NextResponse.json({ error: 'Не удалось обновить слот' }, { status: 500 });
        }

        console.log(`[API SLOTS PATCH] Success: Slot ${slotId} updated to ${shiftCode}`);
        return NextResponse.json(result.rows[0]);
    } catch (error) {
        console.error('[API SLOTS PATCH Error]', error);
        return NextResponse.json({ error: 'Внутренняя ошибка сервера' }, { status: 500 });
    }
}

export async function DELETE(request: NextRequest) {
    const user = await getUserFromRequest(request);
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

        // Проверяем, что слот принадлежит пользователю
        const checkResult = await pool.query(
            'SELECT user_id FROM shifts WHERE id = $1',
            [slotId]
        );
        
        if (checkResult.rowCount === 0) {
            return NextResponse.json({ error: 'Слот не найден' }, { status: 404 });
        }
        
        const slot = checkResult.rows[0];
        
        // Пользователь может удалять только свои слоты
        if (slot.user_id !== user.id) {
            return NextResponse.json({ error: 'Вы можете удалять только свои слоты' }, { status: 403 });
        }

        const result = await pool.query(
          `DELETE FROM shifts WHERE id = $1 AND user_id = $2`,
          [parseInt(slotId, 10), user.id]
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

// OPTIONS для CORS
export async function OPTIONS() {
    return new NextResponse(null, {
        status: 200,
        headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PATCH, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400',
        },
    });
}