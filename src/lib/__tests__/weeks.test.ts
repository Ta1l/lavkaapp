

import { getWeekStart, getWeekDays } from '../weeks';

describe('Week Calculations', () => {
  const mockDate = (dateString: string) => {
    const RealDate = Date;
    const mockDate = new RealDate(dateString);
    
    // @ts-ignore - we're intentionally mocking Date
    global.Date = class extends RealDate {
      constructor(value?: number | string | Date) {
        if (value === undefined) {
          super(mockDate);
        } else {
          super(value);
        }
      }

      static now() {
        return mockDate.getTime();
      }
    };
  };

  afterEach(() => {
    // Restore the real Date after each test
    global.Date = Date;
  });

  test('getWeekStart returns correct Monday for mid-week date', () => {
    mockDate('2025-07-08T12:00:00Z'); // Tuesday
    const result = getWeekStart(new Date());
    // Check that it's a Monday
    expect(result.getUTCDay()).toBe(1); // Monday is 1
    // Check that it's July 7
    expect(result.getUTCDate()).toBe(7);
    expect(result.getUTCMonth()).toBe(6); // July is 6 (0-based)
    expect(result.getUTCFullYear()).toBe(2025);
  });

  test('getWeekDays returns correct week range for offset 0', () => {
    mockDate('2025-07-08T12:00:00Z');
    const monday = getWeekStart(new Date());
    const days = getWeekDays(monday);

    expect(days).toHaveLength(7);
    expect(days[0].dayName).toBe('понедельник');
    expect(days[0].dayOfMonth).toBe('7');
    expect(days[6].dayName).toBe('воскресенье');
    expect(days[6].dayOfMonth).toBe('13');

    // Verify dates are in correct sequence
    const dates = days.map(d => new Date(d.date));
    dates.forEach((date, i) => {
      expect(date.getUTCDate()).toBe(7 + i);
      expect(date.getUTCMonth()).toBe(6); // July
      expect(date.getUTCFullYear()).toBe(2025);
    });
  });

  test('getWeekDays returns correct week range for offset 1', () => {
    mockDate('2025-07-08T12:00:00Z');
    const monday = getWeekStart(new Date());
    const nextMonday = new Date(monday);
    nextMonday.setUTCDate(monday.getUTCDate() + 7);
    const days = getWeekDays(nextMonday);

    expect(days).toHaveLength(7);
    expect(days[0].dayName).toBe('понедельник');
    expect(days[0].dayOfMonth).toBe('14');
    expect(days[6].dayName).toBe('воскресенье');
    expect(days[6].dayOfMonth).toBe('20');

    // Verify dates are in correct sequence
    const dates = days.map(d => new Date(d.date));
    dates.forEach((date, i) => {
      expect(date.getUTCDate()).toBe(14 + i);
      expect(date.getUTCMonth()).toBe(6); // July
      expect(date.getUTCFullYear()).toBe(2025);
    });
  });

  test('Week boundary transition at midnight', () => {
    // Test Sunday night
    mockDate('2025-07-13T23:59:59Z');
    let monday = getWeekStart(new Date());
    expect(monday.getUTCDate()).toBe(7);
    expect(monday.getUTCMonth()).toBe(6); // July
    expect(monday.getUTCFullYear()).toBe(2025);

    // Test Monday morning
    mockDate('2025-07-14T00:00:00Z');
    monday = getWeekStart(new Date());
    expect(monday.getUTCDate()).toBe(14);
    expect(monday.getUTCMonth()).toBe(6); // July
    expect(monday.getUTCFullYear()).toBe(2025);
  });
}); 