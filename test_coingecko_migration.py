#!/usr/bin/env python3
"""
Тестирование миграции на CoinGecko API
Проверяет работу всех основных функций после перехода с ccxt/Binance на CoinGecko
"""

import sys
import os
import requests
import json
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoints():
    """Тестирование API эндпоинтов"""
    base_url = "http://127.0.0.1:5000"
    
    print("Тестирование API эндпоинтов с CoinGecko...")
    
    # Тест 1: Health check
    print("\n1. Проверка health эндпоинта...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Health check: {data.get('status', 'unknown')}")
            print(f"   Версия: {data.get('version', 'unknown')}")
        else:
            print(f"[ERROR] Health check failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Health check error: {str(e)}")
    
    # Тест 2: Market data
    print("\n2. Проверка market-data эндпоинта...")
    try:
        response = requests.get(f"{base_url}/api/market-data", timeout=15)
        if response.status_code == 200:
            data = response.json()
            market_data = data.get('market_data', [])
            print(f"[OK] Market data получен: {len(market_data)} криптовалют")
            if market_data:
                btc_data = next((item for item in market_data if 'BTC' in item.get('symbol', '')), None)
                if btc_data:
                    print(f"   BTC цена: ${btc_data.get('price', 'N/A')}")
                    print(f"   BTC изменение: {btc_data.get('change_24h', 'N/A')}%")
        else:
            print(f"[ERROR] Market data failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Market data error: {str(e)}")
    
    # Тест 3: React market data (новый эндпоинт)
    print("\n3. Проверка react/market-data эндпоинта...")
    try:
        response = requests.get(f"{base_url}/api/react/market-data", timeout=15)
        if response.status_code == 200:
            data = response.json()
            market_data = data.get('data', [])
            print(f"[OK] React market data получен: {len(market_data)} криптовалют")
            if market_data:
                print(f"   Формат данных для React компонентов готов")
        else:
            print(f"[ERROR] React market data failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] React market data error: {str(e)}")
    
    # Тест 4: Crypto prices
    print("\n4. Проверка crypto-prices эндпоинта...")
    try:
        response = requests.get(f"{base_url}/api/crypto-prices", timeout=15)
        if response.status_code == 200:
            data = response.json()
            prices = data.get('prices', {})
            print(f"[OK] Crypto prices получены: {len(prices)} криптовалют")
            if 'bitcoin' in prices:
                print(f"   Bitcoin цена: ${prices['bitcoin'].get('usd', 'N/A')}")
        else:
            print(f"❌ Crypto prices failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Crypto prices error: {str(e)}")
    
    # Тест 5: Realtime prices
    print("\n5. Проверка realtime-prices эндпоинта...")
    try:
        response = requests.get(f"{base_url}/api/realtime-prices", timeout=15)
        if response.status_code == 200:
            data = response.json()
            prices = data.get('prices', [])
            print(f"✅ Realtime prices получены: {len(prices)} криптовалют")
        else:
            print(f"❌ Realtime prices failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Realtime prices error: {str(e)}")

def test_crypto_service():
    """Тестирование CryptoService напрямую"""
    print("\n🔧 Тестирование CryptoService...")
    
    try:
        from app.services.crypto_service import CryptoService
        
        crypto_service = CryptoService()
        
        # Тест получения рыночных данных
        print("\n1. Тестирование get_market_data...")
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        market_data = crypto_service.get_market_data(symbols)
        
        if market_data:
            print(f"✅ Получены данные для {len(market_data)} символов")
            for data in market_data[:2]:  # Показать первые 2
                print(f"   {data.get('symbol', 'N/A')}: ${data.get('price', 'N/A')} ({data.get('change_24h', 'N/A')}%)")
        else:
            print("❌ Не удалось получить рыночные данные")
        
        # Тест получения текущей цены
        print("\n2. Тестирование get_current_price...")
        btc_price = crypto_service.get_current_price('BTC/USDT')
        if btc_price:
            print(f"✅ BTC цена: ${btc_price}")
        else:
            print("❌ Не удалось получить цену BTC")
        
        # Тест получения исторических данных
        print("\n3. Тестирование get_historical_data...")
        historical_data = crypto_service.get_historical_data('BTC/USDT', 7)
        if not historical_data.empty:
            print(f"✅ Получены исторические данные: {len(historical_data)} записей")
            print(f"   Период: {historical_data.index[0]} - {historical_data.index[-1]}")
        else:
            print("❌ Не удалось получить исторические данные")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования CryptoService: {str(e)}")

def test_analysis_service():
    """Тестирование AnalysisService"""
    print("\n📊 Тестирование AnalysisService...")
    
    try:
        from app.services.analysis_service import AnalysisService
        
        analysis_service = AnalysisService()
        
        # Тест анализа тренда
        print("\n1. Тестирование analyze_trend...")
        trend_result = analysis_service.analyze_trend('BTC-USD', '1w')
        if trend_result.get('analysis_data'):
            analysis_data = trend_result['analysis_data']
            print(f"✅ Анализ тренда выполнен")
            print(f"   Тренд: {analysis_data.get('trend', 'N/A')}")
            print(f"   RSI: {analysis_data.get('rsi', 'N/A')}")
        else:
            print("❌ Не удалось выполнить анализ тренда")
            
    except Exception as e:
        print(f"❌ Ошибка тестирования AnalysisService: {str(e)}")

def main():
    """Основная функция тестирования"""
    print("🚀 Начинаем тестирование миграции на CoinGecko API")
    print("=" * 60)
    
    # Проверяем, что сервер запущен
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        print("✅ Сервер доступен")
    except Exception as e:
        print(f"❌ Сервер недоступен: {str(e)}")
        print("Убедитесь, что приложение запущено: python run.py run")
        return
    
    # Запускаем тесты
    test_api_endpoints()
    test_crypto_service()
    test_analysis_service()
    
    print("\n" + "=" * 60)
    print("🎯 Тестирование завершено!")
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
