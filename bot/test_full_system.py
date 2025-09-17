#!/usr/bin/env python3
# test_full_system.py
"""
Полный набор тестов для проверки работы системы Lavka Bot
Запускать перед демонстрацией для проверки всех компонентов
"""

import asyncio
import aiohttp
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Конфигурация
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://slotworker.ru")
BOT_TOKEN = os.getenv("BOT_TOKEN", "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM")

# Используем реальных пользователей для тестов
TEST_USERS = [
    {"username": "66", "password": "66"},  # Существующий пользователь
    {"username": "7", "password": "7"},    # Существующий пользователь
]

# Цвета для вывода
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
RESET = '\033[0m'


class TestResult:
    """Класс для хранения результатов тестов"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.warnings_list = []
        
    def add_pass(self):
        self.passed += 1
        
    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)
        
    def add_warning(self, warning: str):
        self.warnings += 1
        self.warnings_list.append(warning)
        
    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n{BLUE}{'='*60}")
        print("ИТОГИ ТЕСТИРОВАНИЯ")
        print(f"{'='*60}{RESET}")
        
        if self.failed == 0:
            print(f"{GREEN}✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!{RESET}")
        else:
            print(f"{RED}❌ ЕСТЬ ПРОБЛЕМЫ!{RESET}")
            
        print(f"\nВсего тестов: {total}")
        print(f"{GREEN}Успешно: {self.passed}{RESET}")
        print(f"{RED}Провалено: {self.failed}{RESET}")
        print(f"{YELLOW}Предупреждений: {self.warnings}{RESET}")
        
        if self.errors:
            print(f"\n{RED}ОШИБКИ:{RESET}")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")
                
        if self.warnings_list:
            print(f"\n{YELLOW}ПРЕДУПРЕЖДЕНИЯ:{RESET}")
            for i, warning in enumerate(self.warnings_list, 1):
                print(f"{i}. {warning}")


# Глобальный объект для результатов
results = TestResult()
# Глобальная переменная для API ключей
global_api_keys = {}


async def test_server_availability():
    """Тест 1: Проверка доступности серверов"""
    print(f"\n{MAGENTA}=== Тест 1: Проверка доступности серверов ==={RESET}")
    
    endpoints = [
        (f"{WEBAPP_URL}/", "Главная страница"),
        (f"{WEBAPP_URL}/api/auth/get-token", "Auth API", "POST"),
        ("https://api.telegram.org/bot" + BOT_TOKEN + "/getMe", "Telegram Bot API"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            url = endpoint[0]
            name = endpoint[1]
            method = endpoint[2] if len(endpoint) > 2 else "GET"
            
            try:
                if method == "GET":
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            print(f"{GREEN}✓ {name}: Доступен{RESET}")
                            results.add_pass()
                        elif resp.status == 404 and "health" in url:
                            print(f"{YELLOW}⚠ {name}: Endpoint не существует (404){RESET}")
                            results.add_warning(f"{name} endpoint не реализован")
                        else:
                            print(f"{RED}✗ {name}: Статус {resp.status}{RESET}")
                            results.add_fail(f"{name} вернул статус {resp.status}")
                elif method == "POST":
                    # Для POST endpoints проверяем, что они отвечают
                    async with session.post(url, json={}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status in [200, 400, 401]:  # Ожидаемые статусы
                            print(f"{GREEN}✓ {name}: Доступен (статус {resp.status}){RESET}")
                            results.add_pass()
                        else:
                            print(f"{RED}✗ {name}: Неожиданный статус {resp.status}{RESET}")
                            results.add_fail(f"{name} вернул неожиданный статус {resp.status}")
            except Exception as e:
                print(f"{RED}✗ {name}: Недоступен - {e}{RESET}")
                results.add_fail(f"{name} недоступен: {str(e)}")


async def test_authentication():
    """Тест 2: Аутентификация существующих пользователей"""
    print(f"\n{MAGENTA}=== Тест 2: Аутентификация ==={RESET}")
    
    global global_api_keys
    
    for user in TEST_USERS:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{WEBAPP_URL}/api/auth/get-token",
                    json={"username": user["username"], "password": user["password"]}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        api_key = data.get("apiKey")
                        if api_key:
                            global_api_keys[user["username"]] = api_key
                            print(f"{GREEN}✓ Получен API ключ для {user['username']}: {api_key[:20]}...{RESET}")
                            results.add_pass()
                        else:
                            print(f"{RED}✗ API ключ не найден для {user['username']}{RESET}")
                            results.add_fail(f"API ключ не найден для {user['username']}")
                    else:
                        text = await resp.text()
                        print(f"{RED}✗ Ошибка аутентификации {user['username']}: {text[:100]}{RESET}")
                        results.add_fail(f"Ошибка аутентификации {user['username']}")
        except Exception as e:
            print(f"{RED}✗ Ошибка при аутентификации {user['username']}: {e}{RESET}")
            results.add_fail(f"Ошибка аутентификации {user['username']}: {str(e)}")
    
    return global_api_keys


async def test_shift_operations():
    """Тест 3: Операции со слотами"""
    print(f"\n{MAGENTA}=== Тест 3: Операции со слотами ==={RESET}")
    
    if not global_api_keys:
        print(f"{RED}✗ Нет API ключей для тестирования{RESET}")
        results.add_fail("Нет API ключей для тестирования слотов")
        return
    
    # Тестовые слоты на завтра
    tomorrow = datetime.now() + timedelta(days=1)
    test_date = tomorrow.strftime("%Y-%m-%d")
    
    # Сначала очистим слоты на тестовую дату
    username1 = list(global_api_keys.keys())[0]
    api_key1 = global_api_keys[username1]
    
    print(f"\n{BLUE}Подготовка: Очистка тестовых слотов{RESET}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key1}"},
                params={"date": test_date}
            ) as resp:
                if resp.status == 200:
                    print(f"{GREEN}✓ Тестовая дата очищена{RESET}")
                else:
                    print(f"{YELLOW}⚠ Не удалось очистить тестовую дату{RESET}")
    except Exception as e:
        print(f"{YELLOW}⚠ Ошибка при очистке: {e}{RESET}")
    
    # Тестовые слоты
    test_shifts = [
        {
            "date": test_date,
            "startTime": "09:00",
            "endTime": "13:00",
            "assignToSelf": True
        },
        {
            "date": test_date,
            "startTime": "14:00",
            "endTime": "18:00",
            "assignToSelf": True
        }
    ]
    
    # Тест 3.1: Добавление слотов
    print(f"\n{BLUE}Тест 3.1: Добавление слотов{RESET}")
    created_shifts = []
    
    async with aiohttp.ClientSession() as session:
        for shift in test_shifts:
            try:
                async with session.post(
                    f"{WEBAPP_URL}/api/shifts",
                    headers={"Authorization": f"Bearer {api_key1}"},
                    json=shift
                ) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        created_shifts.append(data)
                        print(f"{GREEN}✓ Слот добавлен: {shift['date']} {shift['startTime']}-{shift['endTime']}{RESET}")
                        results.add_pass()
                    else:
                        text = await resp.text()
                        print(f"{RED}✗ Ошибка добавления слота: {resp.status} - {text[:100]}{RESET}")
                        results.add_fail(f"Ошибка добавления слота: {resp.status}")
            except Exception as e:
                print(f"{RED}✗ Ошибка при добавлении слота: {e}{RESET}")
                results.add_fail(f"Ошибка добавления слота: {str(e)}")
    
    # Тест 3.2: Проверка конфликтов
    if len(global_api_keys) > 1 and created_shifts:
        print(f"\n{BLUE}Тест 3.2: Проверка конфликтов{RESET}")
        username2 = list(global_api_keys.keys())[1]
        api_key2 = global_api_keys[username2]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{WEBAPP_URL}/api/shifts",
                    headers={"Authorization": f"Bearer {api_key2}"},
                    json=test_shifts[0]
                ) as resp:
                    if resp.status == 409:
                        print(f"{GREEN}✓ Конфликт правильно обработан (409){RESET}")
                        results.add_pass()
                    elif resp.status in (200, 201):
                        data = await resp.json()
                        if data.get("user_id") == created_shifts[0].get("user_id"):
                            print(f"{GREEN}✓ Слот остался у первого пользователя{RESET}")
                            results.add_pass()
                        else:
                            print(f"{YELLOW}⚠ Слот был переназначен другому пользователю{RESET}")
                            results.add_warning("Слот переназначен без конфликта")
                    else:
                        print(f"{RED}✗ Неожиданный статус: {resp.status}{RESET}")
                        results.add_fail(f"Неожиданный статус при конфликте: {resp.status}")
        except Exception as e:
            print(f"{RED}✗ Ошибка при проверке конфликта: {e}{RESET}")
            results.add_fail(f"Ошибка проверки конфликта: {str(e)}")
    
    # Тест 3.3: Получение списка слотов
    print(f"\n{BLUE}Тест 3.3: Получение списка слотов{RESET}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key1}"},
                params={"start": test_date, "end": test_date}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"{GREEN}✓ Получено слотов: {len(data)}{RESET}")
                    results.add_pass()
                else:
                    print(f"{RED}✗ Ошибка получения слотов: {resp.status}{RESET}")
                    results.add_fail(f"Ошибка получения слотов: {resp.status}")
    except Exception as e:
        print(f"{RED}✗ Ошибка при получении слотов: {e}{RESET}")
        results.add_fail(f"Ошибка получения слотов: {str(e)}")


async def test_bot_integration():
    """Тест 4: Интеграция с Telegram ботом"""
    print(f"\n{MAGENTA}=== Тест 4: Проверка Telegram бота ==={RESET}")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Проверка информации о боте
            async with session.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("ok"):
                        bot_info = data.get("result", {})
                        print(f"{GREEN}✓ Бот активен: @{bot_info.get('username')}{RESET}")
                        results.add_pass()
                else:
                    print(f"{RED}✗ Ошибка проверки бота: {resp.status}{RESET}")
                    results.add_fail(f"Ошибка проверки бота: {resp.status}")
                    
            # Проверка webhook (если используется)
            async with session.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getWebhookInfo"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    webhook_url = data.get("result", {}).get("url", "")
                    if webhook_url:
                        print(f"{YELLOW}⚠ Бот использует webhook: {webhook_url}{RESET}")
                        results.add_warning("Бот использует webhook вместо polling")
                    else:
                        print(f"{GREEN}✓ Бот использует polling{RESET}")
                        results.add_pass()
                        
    except Exception as e:
        print(f"{RED}✗ Ошибка при проверке бота: {e}{RESET}")
        results.add_fail(f"Ошибка проверки бота: {str(e)}")


async def test_performance():
    """Тест 5: Проверка производительности"""
    print(f"\n{MAGENTA}=== Тест 5: Проверка производительности ==={RESET}")
    
    endpoints = [
        (f"{WEBAPP_URL}/", "Главная страница"),
        (f"{WEBAPP_URL}/api/shifts", "API слотов"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for url, name in endpoints:
            try:
                start_time = datetime.now()
                
                # Для API используем авторизацию если есть ключи
                headers = {}
                if "api" in url and global_api_keys:
                    api_key = list(global_api_keys.values())[0]
                    headers = {"Authorization": f"Bearer {api_key}"}
                
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    if duration < 1:
                        print(f"{GREEN}✓ {name}: {duration:.2f}с (отлично){RESET}")
                        results.add_pass()
                    elif duration < 3:
                        print(f"{YELLOW}⚠ {name}: {duration:.2f}с (медленно){RESET}")
                        results.add_warning(f"{name} отвечает медленно: {duration:.2f}с")
                    else:
                        print(f"{RED}✗ {name}: {duration:.2f}с (очень медленно){RESET}")
                        results.add_fail(f"{name} отвечает очень медленно: {duration:.2f}с")
                        
            except asyncio.TimeoutError:
                print(f"{RED}✗ {name}: Timeout (>10с){RESET}")
                results.add_fail(f"{name} timeout")
            except Exception as e:
                print(f"{RED}✗ {name}: Ошибка - {e}{RESET}")
                results.add_fail(f"{name} ошибка: {str(e)}")


async def test_security():
    """Тест 6: Базовые проверки безопасности"""
    print(f"\n{MAGENTA}=== Тест 6: Проверка безопасности ==={RESET}")
    
    # Тест 6.1: Доступ без авторизации
    print(f"\n{BLUE}Тест 6.1: Доступ без авторизации{RESET}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/shifts",
                json={"date": "2025-09-20", "startTime": "10:00", "endTime": "14:00"}
            ) as resp:
                if resp.status == 401:
                    print(f"{GREEN}✓ Неавторизованный доступ правильно заблокирован{RESET}")
                    results.add_pass()
                else:
                    print(f"{RED}✗ Неавторизованный доступ не заблокирован: {resp.status}{RESET}")
                    results.add_fail(f"Неавторизованный доступ не заблокирован")
    except Exception as e:
        print(f"{RED}✗ Ошибка при проверке безопасности: {e}{RESET}")
        results.add_fail(f"Ошибка проверки безопасности: {str(e)}")
    
    # Тест 6.2: Невалидный API ключ
    print(f"\n{BLUE}Тест 6.2: Невалидный API ключ{RESET}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": "Bearer invalid_key_12345"}
            ) as resp:
                if resp.status == 401:
                    print(f"{GREEN}✓ Невалидный ключ правильно отклонен{RESET}")
                    results.add_pass()
                else:
                    print(f"{RED}✗ Невалидный ключ не отклонен: {resp.status}{RESET}")
                    results.add_fail(f"Невалидный ключ не отклонен")
    except Exception as e:
        print(f"{RED}✗ Ошибка при проверке ключа: {e}{RESET}")
        results.add_fail(f"Ошибка проверки ключа: {str(e)}")


async def run_all_tests():
    """Запуск всех тестов"""
    print(f"{BLUE}{'='*60}")
    print(f"КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ LAVKA BOT")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Сервер: {WEBAPP_URL}")
    print(f"{'='*60}{RESET}")
    
    # Запуск тестов
    await test_server_availability()
    await test_authentication()
    await test_shift_operations()
    await test_bot_integration()
    await test_performance()
    await test_security()
    
    # Вывод результатов
    results.print_summary()
    
    # Рекомендации
    print(f"\n{BLUE}РЕКОМЕНДАЦИИ ДЛЯ ДЕМОНСТРАЦИИ:{RESET}")
    if results.failed == 0:
        print(f"{GREEN}✅ Система готова к демонстрации!{RESET}")
        print("\nПорядок демонстрации:")
        print("1. Покажите вход на сайт slotworker.ru")
        print("2. Продемонстрируйте вход в бота @lavkaappbot")
        print("3. Загрузите тестовые скриншоты со слотами")
        print("4. Покажите, как бот распознает слоты")
        print("5. Подтвердите загрузку и покажите слоты на сайте")
        print("\nПодготовьте:")
        print("• Скриншоты с расписанием слотов")
        print("• Логин и пароль для демо (например, пользователь '66')")
        print("• Откройте сайт и бота заранее")
    else:
        print(f"{RED}⚠️ ВНИМАНИЕ: Есть проблемы, которые нужно исправить!{RESET}")
        print("\nКритические проблемы:")
        for error in results.errors[:5]:
            print(f"  - {error}")
    
    # Статус компонентов
    print(f"\n{BLUE}СТАТУС КОМПОНЕНТОВ:{RESET}")
    components = {
        "Веб-сайт": any("Главная страница" in str(e) for e in results.errors) == False,
        "API аутентификации": len(global_api_keys) > 0,
        "API слотов": any("слотов" in str(e) for e in results.errors) == False,
        "Telegram бот": any("Бот активен" in str(e) for e in results.errors) == False,
        "Безопасность": any("безопасности" in str(e) for e in results.errors) == False,
    }
    
    for component, status in components.items():
        if status:
            print(f"  {GREEN}✓ {component}{RESET}")
        else:
            print(f"  {RED}✗ {component}{RESET}")
    
    # Проверка готовности
    all_ready = all(components.values())
    print(f"\n{BLUE}ГОТОВНОСТЬ К ДЕМОНСТРАЦИИ:{RESET}")
    if all_ready:
        print(f"{GREEN}✅ СИСТЕМА ПОЛНОСТЬЮ ГОТОВА!{RESET}")
    else:
        print(f"{YELLOW}⚠️ Требуется устранить проблемы перед демонстрацией{RESET}")


async def quick_health_check():
    """Быстрая проверка здоровья системы"""
    print(f"\n{BLUE}БЫСТРАЯ ПРОВЕРКА СИСТЕМЫ:{RESET}")
    
    checks = {
        "Веб-сайт": f"{WEBAPP_URL}/",
        "API": f"{WEBAPP_URL}/api/auth/get-token",
        "Telegram бот": f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    }
    
    all_ok = True
    async with aiohttp.ClientSession() as session:
        for name, url in checks.items():
            try:
                if "auth/get-token" in url:
                    # Для POST endpoint
                    async with session.post(url, json={}, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status in [200, 400, 401]:
                            print(f"  {GREEN}✓ {name}{RESET}")
                        else:
                            print(f"  {RED}✗ {name} (статус {resp.status}){RESET}")
                            all_ok = False
                else:
                    # Для GET endpoints
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            print(f"  {GREEN}✓ {name}{RESET}")
                        else:
                            print(f"  {RED}✗ {name} (статус {resp.status}){RESET}")
                            all_ok = False
            except Exception:
                print(f"  {RED}✗ {name} (недоступен){RESET}")
                all_ok = False
    
    return all_ok


async def main():
    """Главная функция"""
    # Проверка аргументов командной строки
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Быстрая проверка
            all_ok = await quick_health_check()
            if all_ok:
                print(f"\n{GREEN}✅ Все системы работают!{RESET}")
                sys.exit(0)
            else:
                print(f"\n{RED}❌ Есть проблемы с системой!{RESET}")
                sys.exit(1)
        elif sys.argv[1] == "--help":
            print("Использование:")
            print("  python test_full_system.py         - Полное тестирование")
            print("  python test_full_system.py --quick - Быстрая проверка")
            print("  python test_full_system.py --help  - Эта справка")
            sys.exit(0)
    
    # Полное тестирование
    await run_all_tests()


if __name__ == "__main__":
    # Проверка версии Python
    if sys.version_info < (3, 7):
        print(f"{RED}❌ Требуется Python 3.7 или выше{RESET}")
        sys.exit(1)
    
    # Запуск тестов
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️ Тестирование прервано пользователем{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}❌ Критическая ошибка: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)