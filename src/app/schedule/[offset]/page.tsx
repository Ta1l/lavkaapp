// src/app/schedule/[offset]/page.tsx

import { getCalendarWeeks } from '@/lib/dateUtils';
import { Shift, Day, User } from '@/types/shifts';
import ScheduleClientComponent from '@/components/ScheduleClientComponent';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { notFound } from 'next/navigation';
import { cookies } from 'next/headers';

interface PageProps {
    params: { offset: string; };
    searchParams: { [key: string]: string | string[] | undefined };
}

function getCurrentUserFromCookie(): User | null {
    const sessionCookie = cookies().get('auth-session');
    if (!sessionCookie) return null;
    try {
        return JSON.parse(sessionCookie.value);
    } catch {
        return null;
    }
}

async function getWeekData(offset: number, targetUserId?: number): Promise<Day[]> {
    console.log('[getWeekData] Начало загрузки данных...');
    const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
    let targetWeekTemplate = offset === 0 ? mainWeek : nextWeek;

    if (!targetUserId) {
        console.warn("[getWeekData] Нет targetUserId, возвращаем пустую неделю.");
        return targetWeekTemplate;
    }

    // [ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ] Собираем URL правильно, используя переменную окружения
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';
    const apiUrl = new URL(`${baseUrl}/api/shifts`);
    apiUrl.searchParams.append('offset', String(offset));
    apiUrl.searchParams.append('userId', String(targetUserId));
    
    console.log(`[getWeekData] Запрос на API: ${apiUrl.toString()}`);

    try {
        const response = await fetch(apiUrl.toString(), { 
            cache: 'no-store',
            headers: {
                Cookie: cookies().toString(),
            },
        });

        console.log(`[getWeekData] Ответ от API получен, статус: ${response.status}`);
        if (!response.ok) {
            throw new Error(`API вернул ошибку: ${response.status}`);
        }
        
        const shifts: Shift[] = await response.json();
        console.log(`[getWeekData] Получено ${shifts.length} смен из API.`);

        return targetWeekTemplate.map(day => {
            const dayShifts = shifts.filter(shift => new Date(shift.shift_date).toDateString() === day.date.toDateString());
            return {
                ...day,
                slots: dayShifts.map(shift => ({
                    id: shift.id,
                    startTime: shift.shift_code.split('-')[0],
                    endTime: shift.shift_code.split('-')[1],
                    status: shift.status,
                    user_id: shift.user_id,
                    userName: shift.full_name || shift.username,
                }))
            };
        });
    } catch (error) {
        console.error(`[getWeekData] КРИТИЧЕСКАЯ ОШИБКА при получении данных:`, error);
        return targetWeekTemplate; // Возвращаем пустую структуру в случае сбоя
    }
}

export default async function SchedulePage({ params, searchParams }: PageProps) {
    console.log('\n--- [СЕРВЕР] ЗАГРУЗКА СТРАНИЦЫ РАСПИСАНИЯ ---');
    const currentUser = getCurrentUserFromCookie();
    console.log('[СЕРВЕР] Текущий пользователь из cookie:', currentUser);
    
    const offset = parseInt(params.offset);
    if (isNaN(offset) || ![0, 1].includes(offset)) {
        return notFound();
    }
    
    const viewedUserIdParam = searchParams.user;
    let targetUserId: number | undefined;

    if (viewedUserIdParam && typeof viewedUserIdParam === 'string') {
        targetUserId = parseInt(viewedUserIdParam, 10);
    } else if (currentUser) {
        targetUserId = currentUser.id;
    }
    
    console.log('[СЕРВЕР] ID пользователя для загрузки данных:', targetUserId);
    const isOwner = !viewedUserIdParam || (currentUser?.id === targetUserId);
    
    const weekDays = await getWeekData(offset, targetUserId);
    console.log('--- [СЕРВЕР] ЗАГРУЗКА СТРАНИЦЫ ЗАВЕРШЕНА ---\n');

    return (
        <ErrorBoundary fallback={<p>Произошла ошибка при загрузке расписания.</p>}>
            <ScheduleClientComponent
                initialWeekDays={weekDays}
                initialOffset={offset}
                currentUser={currentUser}
                isOwner={isOwner}
            />
        </ErrorBoundary>
    );
}

export const dynamic = 'force-dynamic';