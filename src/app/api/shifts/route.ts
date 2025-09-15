// src/app/api/shifts/route.ts
// Добавьте в начало файла после импортов:

// Добавьте OPTIONS для CORS
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
      status: 200,
      headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          'Access-Control-Max-Age': '86400',
      },
  });
}

// Обновите функцию POST с логированием в самом начале:
export async function POST(request: NextRequest) {
console.log('[POST /api/shifts] ========== REQUEST START ==========');
console.log('[POST /api/shifts] Method:', request.method);
console.log('[POST /api/shifts] URL:', request.url);
console.log('[POST /api/shifts] Headers:', Object.fromEntries(request.headers.entries()));

try {
  // Проверяем тело запроса
  const body = await request.json();
  console.log('[POST /api/shifts] Body:', body);
  
  // --- ИЗМЕНЕНИЕ: Используем новую функцию аутентификации ---
  const currentUser = await getUserFromRequest(request);
  console.log('[POST /api/shifts] Current user:', currentUser);

  if (!currentUser) {
    console.log('[POST /api/shifts] Unauthorized - no user found');
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { date, startTime, endTime, assignToSelf } = body;
  if (!date || !startTime || !endTime) {
    console.log('[POST /api/shifts] Missing fields:', { date, startTime, endTime });
    return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
  }

  // Остальной код без изменений...
  const dateStr = normalizeDateString(date);
  if (!dateStr) {
    return NextResponse.json({ error: 'Invalid date' }, { status: 400 });
  }
  
  console.log('[POST /api/shifts] Processing shift:', { dateStr, startTime, endTime, assignToSelf });
  
  const shiftCode = `${startTime}-${endTime}`;
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const existing = await client.query(
      'SELECT * FROM shifts WHERE shift_date = $1 AND shift_code = $2',
      [dateStr, shiftCode]
    );

    if ((existing.rowCount ?? 0) > 0) {
      await client.query('COMMIT');
      console.log('[POST /api/shifts] Shift already exists:', existing.rows[0]);
      return NextResponse.json(existing.rows[0]);
    }
    
    const userId = assignToSelf ? currentUser.id : null;
    const status = assignToSelf ? 'pending' : 'available';
    const insert = await client.query(
      `INSERT INTO shifts (shift_date, day_of_week, shift_code, status, user_id)
       VALUES ($1, EXTRACT(ISODOW FROM $1::date), $2, $3, $4)
       RETURNING *`,
      [dateStr, shiftCode, status, userId]
    );
    await client.query('COMMIT');
    
    console.log('[POST /api/shifts] Shift created:', insert.rows[0]);
    return NextResponse.json(insert.rows[0], { status: 201 });

  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
  }
} catch (err) {
  console.error('[POST /api/shifts] Error:', err);
  return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
}
}