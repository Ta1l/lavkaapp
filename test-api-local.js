// test-api-local.js
const API_KEY = '139ae1206c7992621c899f73f6e99bb0'; // Ваш API ключ из теста

async function testAPI() {
    console.log('Testing API locally...\n');
    
    // Test 1: GET request
    console.log('1. Testing GET /api/shifts');
    try {
        const getResponse = await fetch('http://localhost:3000/api/shifts', {
            headers: {
                'Authorization': `Bearer ${API_KEY}`
            }
        });
        console.log('GET Status:', getResponse.status);
        const getData = await getResponse.text();
        console.log('GET Response:', getData.substring(0, 200));
    } catch (e) {
        console.error('GET Error:', e.message);
    }
    
    // Test 2: POST request
    console.log('\n2. Testing POST /api/shifts');
    try {
        const postResponse = await fetch('http://localhost:3000/api/shifts', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                date: '2025-09-16',
                startTime: '10:00',
                endTime: '18:00',
                assignToSelf: true
            })
        });
        console.log('POST Status:', postResponse.status);
        const postData = await postResponse.text();
        console.log('POST Response:', postData.substring(0, 200));
    } catch (e) {
        console.error('POST Error:', e.message);
    }
}

testAPI();