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
import random
import string

# Конфигурация
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://slotworker.ru")
BOT_TOKEN = os.getenv("BOT_TOKEN", "8457174750:AAHAz3tAjrUkEPZHX1mJvuDUJj7YkzbhlMM")

# Тестовые пользователи
TEST_USERS = [
    {"username": "test_user_1", "password": "test123"},
    {"username": "test_user_2", "password": "test123"},
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


async def test_server_availability():
    """Тест 1: Проверка доступности серверов"""
    print(f"\n{MAGENTA}=== Тест 1: Проверка доступности серверов ==={RESET}")
    
    endpoints = [
        (f"{WEBAPP_URL}/", "Главная страница"),
        (f"{WEBAPP_URL}/api/health", "Health check API"),
        ("https://api.telegram.org/bot" + BOT_TOKEN + "/getMe", "Telegram Bot API"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for url, name in endpoints:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        print(f"{GREEN}✓ {name}: Доступен{RESET}")
                        results.add_pass()
                    else:
                        print(f"{RED}✗ {name}: Статус {resp.status}{RESET}")
                        results.add_fail(f"{name} вернул статус {resp.status}")
            except Exception as e:
                print(f"{RED}✗ {name}: Недоступен - {e}{RESET}")
                results.add_fail(f"{name} недоступен: {str(e)}")


async def test_user_registration():
    """Тест 2: Регистрация пользователей"""
    print(f"\n{MAGENTA}=== Тест 2: Регистрация пользователей ==={RESET}")
    
    for user in TEST_USERS:
        try:
            async with aiohttp.ClientSession() as session:
                # Сначала пробуем войти
                async with session.post(
                    f"{WEBAPP_URL}/api/auth/get-token",
                    json={"username": user["username"], "password": user["password"]}
                ) as resp:
                    if resp.status == 200:
                        print(f"{GREEN}✓ Пользователь {user['username']} уже существует{RESET}")
                        results.add_pass()
                    else:
                        # Пробуем зарегистрировать
                        async with session.post(
                            f"{WEBAPP_URL}/api/auth/register",
                            json={
                                "username": user["username"],
                                "password": user["password"],
                                "fullName": f"Test User {user['username']}"
                            }
                        ) as reg_resp:
                            if reg_resp.status in (200, 201):
                                print(f"{GREEN}✓ Пользователь {user['username']} зарегистрирован{RESET}")
                                results.add_pass()
                            else:
                                text = await reg_resp.text()
                                print(f"{YELLOW}⚠ Не удалось зарегистрировать {user['username']}: {text[:100]}{RESET}")
                                results.add_warning(f"Регистрация {user['username']} не удалась")
        except Exception as e:
            print(f"{RED}✗ Ошибка при регистрации {user['username']}: {e}{RESET}")
            results.add_fail(f"Ошибка регистрации {user['username']}: {str(e)}")


async def test_authentication():
    """Тест 3: Аутентификация и получение API ключей"""
    print(f"\n{MAGENTA}=== Тест 3: Аутентификация ==={RESET}")
    
    api_keys = {}
    
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
                            api_keys[user["username"]] = api_key
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
    
    return api_keys


async def test_shift_operations(api_keys: Dict[str, str]):
    """Тест 4: Операции со слотами"""
    print(f"\n{MAGENTA}=== Тест 4: Операции со слотами ==={RESET}")
    
    if not api_keys:
        print(f"{RED}✗ Нет API ключей для тестирования{RESET}")
        results.add_fail("Нет API ключей для тестирования слотов")
        return
    
    # Тестовые слоты
    tomorrow = datetime.now() + timedelta(days=1)
    test_shifts = [
        {
            "date": tomorrow.strftime("%Y-%m-%d"),
            "startTime": "09:00",
            "endTime": "13:00",
            "assignToSelf": True
        },
        {
            "date": tomorrow.strftime("%Y-%m-%d"),
            "startTime": "14:00",
            "endTime": "18:00",
            "assignToSelf": True
        }
    ]
    
    # Тест 4.1: Добавление слотов первым пользователем
    print(f"\n{BLUE}Тест 4.1: Добавление слотов{RESET}")
    username1 = list(api_keys.keys())[0]
    api_key1 = api_keys[username1]
    
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
                        print(f"{RED}✗ Ошибка добавления слота: {text[:100]}{RESET}")
                        results.add_fail(f"Ошибка добавления слота: {resp.status}")
            except Exception as e:
                print(f"{RED}✗ Ошибка при добавлении слота: {e}{RESET}")
                results.add_fail(f"Ошибка добавления слота: {str(e)}")
    
    # Тест 4.2: Попытка взять занятый слот другим пользователем
    if len(api_keys) > 1 and created_shifts:
        print(f"\n{BLUE}Тест 4.2: Проверка конфликтов{RESET}")
        username2 = list(api_keys.keys())[1]
        api_key2 = api_keys[username2]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{WEBAPP_URL}/api/shifts",
                    headers={"Authorization": f"Bearer {api_key2}"},
                    json=test_shifts[0]  # Пробуем взять первый слот
                ) as resp:
                    if resp.status == 409:
                        print(f"{GREEN}✓ Конфликт правильно обработан (409){RESET}")
                        results.add_pass()
                    elif resp.status in (200, 201):
                        print(f"{YELLOW}⚠ Слот был переназначен другому пользователю{RESET}")
                        results.add_warning("Слот переназначен без конфликта")
                    else:
                        print(f"{RED}✗ Неожиданный статус: {resp.status}{RESET}")
                        results.add_fail(f"Неожиданный статус при конфликте: {resp.status}")
        except Exception as e:
            print(f"{RED}✗ Ошибка при проверке конфликта: {e}{RESET}")
            results.add_fail(f"Ошибка проверки конфликта: {str(e)}")
    
    # Тест 4.3: Получение списка слотов
    print(f"\n{BLUE}Тест 4.3: Получение списка слотов{RESET}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key1}"},
                params={"start": tomorrow.strftime("%Y-%m-%d")}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"{GREEN}✓ Получено слотов: {len(data)}{RESET}")
                    results.add_pass()
                    
                    # Проверяем, что наши слоты есть в списке
                    our_slots = [s for s in data if s.get("user_id") == created_shifts[0].get("user_id")]
                    if our_slots:
                        print(f"{GREEN}✓ Найдены наши слоты: {len(our_slots)}{RESET}")
                        results.add_pass()
                    else:
                        print(f"{YELLOW}⚠ Наши слоты не найдены в списке{RESET}")
                        results.add_warning("Слоты не найдены в списке")
                else:
                    print(f"{RED}✗ Ошибка получения слотов: {resp.status}{RESET}")
                    results.add_fail(f"Ошибка получения слотов: {resp.status}")
    except Exception as e:
        print(f"{RED}✗ Ошибка при получении слотов: {e}{RESET}")
        results.add_fail(f"Ошибка получения слотов: {str(e)}")
    
    # Тест 4.4: Удаление слотов
    print(f"\n{BLUE}Тест 4.4: Удаление слотов{RESET}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key1}"},
                params={"date": tomorrow.strftime("%Y-%m-%d")}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"{GREEN}✓ Слоты удалены/освобождены{RESET}")
                    results.add_pass()
                else:
                    print(f"{RED}✗ Ошибка удаления слотов: {resp.status}{RESET}")
                    results.add_fail(f"Ошибка удаления слотов: {resp.status}")
    except Exception as e:
                print(f"{RED}✗ Ошибка при удалении слотов: {e}{RESET}")
        results.add_fail(f"Ошибка удаления слотов: {str(e)}")


async def test_bot_integration(api_keys: Dict[str, str]):
    """Тест 5: Интеграция с Telegram ботом"""
    print(f"\n{MAGENTA}=== Тест 5: Проверка Telegram бота ==={RESET}")
    
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
                        print(f"{RED}✗ Бот не активен{RESET}")
                        results.add_fail("Telegram бот не активен")
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


async def test_database_consistency():
    """Тест 6: Проверка консистентности данных"""
    print(f"\n{MAGENTA}=== Тест 6: Проверка консистентности данных ==={RESET}")
    
    # Этот тест требует прямого доступа к БД или специального API endpoint
    # Здесь мы проверяем через API
    
    if not api_keys:
        print(f"{YELLOW}⚠ Пропуск теста - нет API ключей{RESET}")
        results.add_warning("Тест консистентности пропущен")
        return
        
    api_key = list(api_keys.values())[0]
    
    try:
        async with aiohttp.ClientSession() as session:
            # Получаем слоты на неделю вперед
            start_date = datetime.now()
            end_date = start_date + timedelta(days=7)
            
            async with session.get(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key}"},
                params={
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                }
            ) as resp:
                if resp.status == 200:
                    shifts = await resp.json()
                    
                    # Проверки консистентности
                    issues = []
                    
                    for shift in shifts:
                        # Проверка 1: shift_code соответствует времени
                        shift_code = shift.get("shift_code", "")
                        if "-" in shift_code:
                            times = shift_code.split("-")
                            if len(times) != 2:
                                issues.append(f"Неверный формат shift_code: {shift_code}")
                        
                        # Проверка 2: статус корректный
                        status = shift.get("status", "")
                        valid_statuses = ["pending", "confirmed", "cancelled", "available"]
                        if status not in valid_statuses:
                            issues.append(f"Неверный статус: {status}")
                        
                        # Проверка 3: если есть user_id, статус не должен быть available
                        if shift.get("user_id") and shift.get("status") == "available":
                            issues.append(f"Слот с user_id имеет статус available: {shift}")
                    
                    if issues:
                        print(f"{YELLOW}⚠ Найдены проблемы консистентности:{RESET}")
                        for issue in issues[:5]:  # Показываем первые 5
                            print(f"  - {issue}")
                        results.add_warning(f"Найдено {len(issues)} проблем консистентности")
                    else:
                        print(f"{GREEN}✓ Данные консистентны{RESET}")
                        results.add_pass()
                else:
                    print(f"{RED}✗ Ошибка получения данных: {resp.status}{RESET}")
                    results.add_fail(f"Ошибка получения данных: {resp.status}")
                    
    except Exception as e:
        print(f"{RED}✗ Ошибка при проверке консистентности: {e}{RESET}")
        results.add_fail(f"Ошибка проверки консистентности: {str(e)}")


async def test_performance():
    """Тест 7: Проверка производительности"""
    print(f"\n{MAGENTA}=== Тест 7: Проверка производительности ==={RESET}")
    
    endpoints = [
        (f"{WEBAPP_URL}/", "Главная страница"),
        (f"{WEBAPP_URL}/api/health", "Health API"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for url, name in endpoints:
            try:
                start_time = datetime.now()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
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
    """Тест 8: Базовые проверки безопасности"""
    print(f"\n{MAGENTA}=== Тест 8: Проверка безопасности ==={RESET}")
    
    # Тест 8.1: Доступ без авторизации
    print(f"\n{BLUE}Тест 8.1: Доступ без авторизации{RESET}")
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
    
    # Тест 8.2: Невалидный API ключ
    print(f"\n{BLUE}Тест 8.2: Невалидный API ключ{RESET}")
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


async def test_edge_cases():
    """Тест 9: Граничные случаи"""
    print(f"\n{MAGENTA}=== Тест 9: Проверка граничных случаев ==={RESET}")
    
    if not api_keys:
        print(f"{YELLOW}⚠ Пропуск теста - нет API ключей{RESET}")
        results.add_warning("Тест граничных случаев пропущен")
        return
        
    api_key = list(api_keys.values())[0]
    
    # Тест 9.1: Слот в прошлом
    print(f"\n{BLUE}Тест 9.1: Слот в прошлом{RESET}")
    yesterday = datetime.now() - timedelta(days=1)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "date": yesterday.strftime("%Y-%m-%d"),
                    "startTime": "10:00",
                    "endTime": "14:00",
                    "assignToSelf": True
                }
            ) as resp:
                if resp.status in (400, 422):
                    print(f"{GREEN}✓ Слот в прошлом правильно отклонен{RESET}")
                    results.add_pass()
                elif resp.status in (200, 201):
                    print(f"{YELLOW}⚠ Система позволяет создавать слоты в прошлом{RESET}")
                    results.add_warning("Можно создавать слоты в прошлом")
                else:
                    print(f"{RED}✗ Неожиданный ответ: {resp.status}{RESET}")
                    results.add_fail(f"Неожиданный ответ для слота в прошлом: {resp.status}")
    except Exception as e:
        print(f"{RED}✗ Ошибка: {e}{RESET}")
        results.add_fail(f"Ошибка теста слота в прошлом: {str(e)}")
    
    # Тест 9.2: Невалидное время
    print(f"\n{BLUE}Тест 9.2: Невалидное время{RESET}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{WEBAPP_URL}/api/shifts",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "date": "2025-09-20",
                    "startTime": "25:00",  # Невалидное время
                    "endTime": "14:00",
                    "assignToSelf": True
                }
            ) as resp:
                if resp.status in (400, 422):
                    print(f"{GREEN}✓ Невалидное время правильно отклонено{RESET}")
                    results.add_pass()
                else:
                    print(f"{YELLOW}⚠ Невалидное время не отклонено: {resp.status}{RESET}")
                    results.add_warning("Невалидное время не отклонено")
    except Exception as e:
        print(f"{RED}✗ Ошибка: {e}{RESET}")
        results.add_fail(f"Ошибка теста невалидного времени: {str(e)}")


async def run_all_tests():
    """Запуск всех тестов"""
    print(f"{BLUE}{'='*60}")
    print(f"КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ LAVKA BOT")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Сервер: {WEBAPP_URL}")
    print(f"{'='*60}{RESET}")
    
    # Запуск тестов
    await test_server_availability()
    await test_user_registration()
    api_keys = await test_authentication()
    await test_shift_operations(api_keys)
    await test_bot_integration(api_keys)
    await test_database_consistency()
    await test_performance()
    await test_security()
    await test_edge_cases()
    
    # Вывод результатов
    results.print_summary()
    
    # Рекомендации
    print(f"\n{BLUE}РЕКОМЕНДАЦИИ ДЛЯ ДЕМОНСТРАЦИИ:{RESET}")
    if results.failed == 0:
        print(f"{GREEN}✅ Система готова к демонстрации!{RESET}")
        print("\nПорядок демонстрации:")
        print("1. Покажите регистрацию на сайте")
        print("2. Продемонстрируйте вход в бота")
        print("3. Загрузите тестовые скриншоты")
        print("4. Покажите распознавание слотов")
                print("5. Продемонстрируйте загрузку слотов на сайт")
        print("6. Покажите слоты на веб-интерфейсе")
        print("7. Продемонстрируйте конфликт слотов (опционально)")
    else:
        print(f"{RED}⚠️ ВНИМАНИЕ: Есть проблемы, которые нужно исправить!{RESET}")
        print("\nКритические проблемы для исправления:")
        for error in results.errors[:5]:  # Показываем первые 5 ошибок
            print(f"  - {error}")
            
    # Дополнительные рекомендации
    print(f"\n{BLUE}ПОДГОТОВКА К ДЕМОНСТРАЦИИ:{RESET}")
    print("1. Подготовьте тестовые скриншоты со слотами")
    print("2. Убедитесь, что бот запущен: python bot.py")
    print("3. Проверьте, что сайт доступен: " + WEBAPP_URL)
    print("4. Создайте тестового пользователя для демо")
    print("5. Очистите старые тестовые данные")
    
    # Контрольный чек-лист
    print(f"\n{BLUE}ЧЕК-ЛИСТ ПЕРЕД ПОКАЗОМ:{RESET}")
    checklist = [
        ("Сервер доступен", results.passed > 0),
        ("API работает", "api_keys" in locals() and len(api_keys) > 0),
        ("Бот активен", any("Бот активен" in str(e) for e in results.errors) == False),
        ("Слоты добавляются", any("добавления слота" in str(e) for e in results.errors) == False),
        ("Безопасность настроена", any("безопасности" in str(e) for e in results.errors) == False),
    ]
    
    for item, status in checklist:
        if status:
            print(f"  {GREEN}✓ {item}{RESET}")
        else:
            print(f"  {RED}✗ {item}{RESET}")


async def quick_health_check():
    """Быстрая проверка здоровья системы"""
    print(f"\n{BLUE}БЫСТРАЯ ПРОВЕРКА:{RESET}")
    
    checks = {
        "Веб-сайт": f"{WEBAPP_URL}/",
        "API": f"{WEBAPP_URL}/api/health",
        "Telegram бот": f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    }
    
    all_ok = True
    async with aiohttp.ClientSession() as session:
        for name, url in checks.items():
            try:
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
    
    # Проверка переменных окружения
    if not WEBAPP_URL:
        print(f"{RED}❌ Не задана переменная WEBAPP_URL{RESET}")
        sys.exit(1)
        
    if not BOT_TOKEN:
        print(f"{RED}❌ Не задан BOT_TOKEN{RESET}")
        sys.exit(1)
    
    # Запуск тестов
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠️ Тестирование прервано пользователем{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}❌ Критическая ошибка: {e}{RESET}")
        sys.exit(1)