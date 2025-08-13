// src/app/schedule/[offset]/page.tsx

import { getCalendarWeeks } from '@/lib/dateUtils';
import { Shift, Day, User } from '@/types/shifts';
import ScheduleClientComponent from '@/components/ScheduleClientComponent';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { notFound } from 'next/navigation';
import { cookies } from 'next/headers';
import { format, addDays } from 'date-fns';

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

// --- ИЗМЕНЕННАЯ ФУНКЦИЯ ---
async function getWeekData(offset: number, targetUserId?: number): Promise<Day[]> {
    const { mainWeek, nextWeek } = getCalendarWeeks(new Date());
    let targetWeekTemplate = offset === 0 ? mainWeek : nextWeek;

    // Определяем даты начала и конца недели для запроса к API
    const startDate = format(targetWeekTemplate[0].date, 'yyyy-MM-dd');
    const endDate = format(addDays(targetWeekTemplate[6].date, 1), 'yyyy-MM-dd'); // +1 день, т.к. API ищет до этой даты, не включая ее

    // Формируем URL для нашего единого API
    const apiUrl = new URL(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'}/api/shifts`);
    apiUrl.searchParams.append('start', startDate);
    apiUrl.searchParams.append('end', endDate);
    if (targetUserId) {
        apiUrl.searchParams.append('userId', String(targetUserId));
    }

    try {
        const response = await fetch(apiUrl.toString(), { 
            cache: 'no-store', // Всегда запрашиваем свежие данные
            headers: {
                Cookie: cookies().toString(),
            },
        });

        if (!response.ok) {
            throw new Error(`API returned status: ${response.status} ${await response.text()}`);
        }
        
        const shifts: Shift[] = await response.json();

        // Распределяем полученные смены по дням недели
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
        console.error(`[Page Data Error] Failed to fetch week data for offset ${offset}:`, error);
        // В случае ошибки возвращаем пустой шаблон недели
        return targetWeekTemplate;
    }
}

export default async function SchedulePage({ params, searchParams }: PageProps) {
    const offset = parseInt(params.offset);
    if (isNaN(offset) || ![0, 1].includes(offset)) {
        return notFound();
    }
    
    const currentUser = getCurrentUserFromCookie();
    const viewedUserIdParam = searchParams.user;
    
    let targetUserId: number | undefined;

    if (viewedUserIdParam && typeof viewedUserIdParam === 'string') {
        targetUserId = parseInt(viewedUserIdParam, 10);
    } else if (currentUser) {
        targetUserId = currentUser.id;
    }

    // Определяем, является ли текущий пользователь владельцем просматриваемого расписания
    const isOwner = !viewedUserIdParam || (currentUser?.id === targetUserId);
    
    const weekDays = await getWeekData(offset, targetUserId);

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

// Говорим Next.js, что страница всегда должна рендериться динамически
export const dynamic = 'force-dynamic';