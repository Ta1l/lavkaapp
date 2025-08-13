// src/types/shifts.ts
export type ShiftStatus = 'pending' | 'confirmed' | 'cancelled' | 'available';

export interface User {
  id: number;
  username: string;
  full_name?: string;
}

export interface Shift {
  id: number;
  user_id: number | null;
  shift_date: string;
  day_of_week: number;
  shift_code: string;
  status: ShiftStatus;
  created_at: string;
  updated_at: string;
  username?: string;
  full_name?: string;
}

export interface TimeSlot {
  id: number;
  startTime: string;
  endTime: string;
  status: ShiftStatus;
  user_id: number | null;
  userName?: string | null;
}

export interface Day {
  date: Date;
  formattedDate: string;
  isToday: boolean;
  slots: TimeSlot[];
}
