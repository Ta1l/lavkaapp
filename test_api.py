#!/usr/bin/env python3
# test_api.py
"""
Тесты для проверки работы API слотов
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import sys

# Конфигурация
WEBAPP_URL = "https://slotworker.ru"
TEST_USERNAME = "66"  # Замените на ваш тестовый логин
TEST_PASSWORD = "66"  # Замените на ваш тестовый пароль

# Цвета для вывода
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


async def test_get_token():
    """Тест 1: Получение API токена"""
    print(f"\n{BLUE}=== Тест 1: Получение API токена ==={RESET}")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{WEBAPP_URL}/api/auth/get-token"
            payload = {
                "username": TEST_USERNAME,
                "password": TEST_PASSWORD
            }
            
            print(f"POST {url}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            async with session.post(url, json=payload) as resp:
                status = resp.status
                text = await resp.text()
                
                print(f"Status: {status}")
                print(f"Response: {text[:200]}...")
                
                if status == 200:
                    data = await resp.json()
                    api_key = data.get("apiKey") or data.get("api_key")
                    if api_key:
                        print(f"{GREEN}✓ Успешно получен API ключ: {api_key[:20]}...{RESET}")
                        return api_key
                    else:
                        print(f"{RED}✗ API ключ не найден в ответе{RESET}")
                        return None
                else:
                    print(f"{RED}✗ Ошибка: статус {status}{RESET}")
                    return None
                    
    except Exception as e:
        print(f"{RED}✗ Ошибка: {e}{RESET}")
        return None


async def test_api_endpoints(api_key: str):
    """Тест 2: Проверка различных вариантов API endpoints"""
    print(f"\n{BLUE}=== Тест 2: Проверка API endpoints ==={RESET}")
    
    endpoints = [
        "/api/shifts",
        "/api/shift",
        "/api/schedule",
        "/api/schedules",
        "/api/slots",
        "/api/slot"
    ]
    
    test_payload = {
        "date": "2025-09-15",
        "startTime": "09:00",
        "endTime": "17:00",
        "assignToSelf": True
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    working_endpoints = []
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            url = f"{WEBAPP_URL}{endpoint}"
            print(f"\nТестирую: {url}")
            
            try:
                async with session.post(
                    url, 
                    json=test_payload, 
                    headers=headers,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    status = resp.status
                    text = await resp.text()
                    
                    print(f"Status: {status}")
                    
                    if status in (200, 201):
                        print(f"{GREEN}✓ Endpoint работает!{RESET}")
                        working_endpoints.append(endpoint)
                    elif status == 307:
                        location = resp.headers.get('Location', 'unknown')
                        print(f"{YELLOW}→ Редирект на: {location}{RESET}")
                    elif status == 404:
                        print(f"{YELLOW}✗ Endpoint не найден{RESET}")
                    elif status == 405:
                        print(f"{YELLOW}✗ Метод не разрешен{RESET}")
                    elif status == 401:
                        print(f"{YELLOW}✗ Не авторизован (проблема с API ключом){RESET}")
                    else:
                        print(f"{RED}✗ Ошибка: {text[:100]}...{RESET}")
                        
            except Exception as e:
                print(f"{RED}✗ Ошибка запроса: {e}{RESET}")
    
    return working_endpoints


async def test_direct_curl_equivalent(api_key: str):
    """Тест 3: Эквивалент curl запроса"""
    print(f"\n{BLUE}=== Тест 3: Прямой запрос (эквивалент curl) ==={RESET}")
    
    url = f"{WEBAPP_URL}/api/shifts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "TestBot/1.0"
    }
    
    # Тестовый слот на завтра
    tomorrow = datetime.now() + timedelta(days=1)
    payload = {
        "date": tomorrow.strftime("%Y-%m-%d"),
        "startTime": "10:00",
        "endTime": "18:00",
        "assignToSelf": True
    }
    
    print(f"URL: {url}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Тест с SSL
        print(f"\n{YELLOW}Тест с SSL проверкой:{RESET}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                print(f"Status: {resp.status}")
                print(f"Headers: {dict(resp.headers)}")
                text = await resp.text()
                print(f"Response: {text[:500]}...")
                
                if resp.status in (200, 201):
                    print(f"{GREEN}✓ Успешно!{RESET}")
                    return True
        
        # Тест без SSL проверки
        print(f"\n{YELLOW}Тест без SSL проверки:{RESET}")
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                print(f"Status: {resp.status}")
                text = await resp.text()
                print(f"Response: {text[:500]}...")
                
    except Exception as e:
        print(f"{RED}✗ Ошибка: {type(e).__name__}: {e}{RESET}")
        return False


async def test_get_shifts(api_key: str):
    """Тест 4: Получение списка слотов"""
    print(f"\n{BLUE}=== Тест 4: Получение списка слотов (GET) ==={RESET}")
    
    url = f"{WEBAPP_URL}/api/shifts"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                status = resp.status
                print(f"GET {url}")
                print(f"Status: {status}")
                
                if status == 200:
                    data = await resp.json()
                    print(f"{GREEN}✓ Получено слотов: {len(data)}{RESET}")
                    if data:
                        print(f"Пример слота: {json.dumps(data[0], indent=2)}")
                else:
                    text = await resp.text()
                    print(f"{RED}✗ Ошибка: {text[:200]}...{RESET}")
                    
    except Exception as e:
        print(f"{RED}✗ Ошибка: {e}{RESET}")


async def test_server_connectivity():
    """Тест 5: Проверка доступности сервера"""
    print(f"\n{BLUE}=== Тест 5: Проверка доступности сервера ==={RESET}")
    
    tests = [
        (f"{WEBAPP_URL}/", "GET", None),
        (f"{WEBAPP_URL}/api", "GET", None),
        (f"{WEBAPP_URL}/api/health", "GET", None),
    ]
    
    async with aiohttp.ClientSession() as session:
        for url, method, data in tests:
            try:
                print(f"\n{method} {url}")
                
                if method == "GET":
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        print(f"Status: {resp.status}")
                        if resp.status == 200:
                            print(f"{GREEN}✓ Доступен{RESET}")
                        else:
                            print(f"{YELLOW}⚠ Статус {resp.status}{RESET}")
                            
            except Exception as e:
                print(f"{RED}✗ Недоступен: {e}{RESET}")


async def main():
    """Основная функция тестирования"""
    print(f"{BLUE}{'='*60}")
    print(f"Тестирование API slotworker.ru")
    print(f"Время: {datetime.now()}")
    print(f"{'='*60}{RESET}")
    
    # Тест 5: Проверка сервера
    await test_server_connectivity()
    
    # Тест 1: Получение токена
    api_key = await test_get_token()
    if not api_key:
        print(f"\n{RED}Не удалось получить API ключ. Проверьте логин/пароль.{RESET}")
        return
    
    # Тест 2: Проверка endpoints
    working_endpoints = await test_api_endpoints(api_key)
    
    # Тест 3: Прямой запрос
    await test_direct_curl_equivalent(api_key)
    
    # Тест 4: GET запрос
    await test_get_shifts(api_key)
    
    # Итоги
    print(f"\n{BLUE}{'='*60}")
    print("ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"{'='*60}{RESET}")
    
    if working_endpoints:
        print(f"{GREEN}✓ Найдены рабочие endpoints: {', '.join(working_endpoints)}{RESET}")
    else:
        print(f"{RED}✗ Рабочие endpoints не найдены{RESET}")
        print(f"\n{YELLOW}Возможные причины:")
        print("1. API не развернут на сервере")
        print("2. Nginx неправильно настроен")
        print("3. Next.js приложение не запущено")
        print("4. Middleware блокирует API запросы{RESET}")


if __name__ == "__main__":
    # Проверяем Python версию
    if sys.version_info < (3, 7):
        print(f"{RED}Требуется Python 3.7 или выше{RESET}")
        sys.exit(1)
    
    # Запускаем тесты
    asyncio.run(main())