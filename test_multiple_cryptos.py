"""
Тест анализа множественных криптовалютных пар
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.analysis_service import AnalysisService, CRYPTO_CATEGORIES
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multiple_cryptos_test.log', encoding='utf-8')
    ]
)

def test_multiple_cryptos():
    """Тестирование анализа множественных криптовалют"""
    
    # Создаем Flask приложение для контекста
    app = create_app()
    
    with app.app_context():
        print("ТЕСТ АНАЛИЗА МНОЖЕСТВЕННЫХ КРИПТОВАЛЮТ")
        print("=" * 60)
        
        # Инициализируем сервис анализа
        analysis_service = AnalysisService()
        
        # Получаем поддерживаемые пары и категории
        supported_pairs = analysis_service.get_supported_crypto_pairs()
        categories = analysis_service.get_crypto_categories()
        
        print(f"\nПоддерживаемые криптовалютные пары: {len(supported_pairs)}")
        for i, pair in enumerate(supported_pairs, 1):
            print(f"  {i:2d}. {pair}")
        
        print(f"\nКатегории криптовалют:")
        for category, pairs in categories.items():
            print(f"  {category.upper()}: {len(pairs)} пар - {', '.join(pairs)}")
        
        # Тестируем анализ категории "major" (топ криптовалюты)
        print(f"\nТестируем анализ категории 'major'...")
        
        try:
            category_results = analysis_service.analyze_crypto_category(
                category='major',
                timeframe='1d',
                limit=3  # Ограничиваем для быстрого тестирования
            )
            
            print(f"\nРЕЗУЛЬТАТЫ АНАЛИЗА КАТЕГОРИИ 'MAJOR':")
            print("-" * 50)
            
            print(f"Категория: {category_results['category']}")
            print(f"Всего пар: {category_results['total_pairs']}")
            print(f"Успешных анализов: {category_results['successful_analyses']}")
            
            # Выводим результаты по каждой паре
            for pair, result in category_results['results'].items():
                print(f"\n{pair}:")
                if 'error' in result:
                    print(f"  Ошибка: {result['error']}")
                else:
                    current_price = result.get('current_price', 0)
                    predicted_price = result.get('predicted_price', 0)
                    confidence = result.get('confidence', 0)
                    
                    if current_price > 0 and predicted_price > 0:
                        change_percent = ((predicted_price - current_price) / current_price) * 100
                        print(f"  Текущая цена: ${current_price:,.2f}")
                        print(f"  Предсказанная цена: ${predicted_price:,.2f}")
                        print(f"  Изменение: {change_percent:+.2f}%")
                        print(f"  Уверенность: {confidence:.2f}")
            
            # Выводим сводку по категории
            summary = category_results.get('summary', {})
            if summary and 'message' not in summary:
                print(f"\nСВОДКА ПО КАТЕГОРИИ:")
                print(f"  Успешных пар: {summary['successful_pairs']}")
                print(f"  Средняя уверенность: {summary['average_confidence']:.2f}")
                print(f"  Среднее изменение цены: {summary['average_price_change']:+.2f}%")
                print(f"  Бычьих пар: {summary['bullish_pairs']}")
                print(f"  Медвежьих пар: {summary['bearish_pairs']}")
                print(f"  Нейтральных пар: {summary['neutral_pairs']}")
                print(f"  Общий тренд категории: {summary['category_trend']}")
            
            print(f"\nАНАЛИЗ КАТЕГОРИИ 'MAJOR' ЗАВЕРШЕН УСПЕШНО!")
            
        except Exception as e:
            print(f"Ошибка при анализе категории: {e}")
            import traceback
            traceback.print_exc()
        
        # Тестируем анализ отдельных пар DeFi
        print(f"\n" + "="*60)
        print(f"Тестируем анализ DeFi токенов...")
        
        try:
            defi_pairs = ['UNI-USD', 'AAVE-USD']  # Ограничиваем для быстрого тестирования
            
            defi_results = analysis_service.analyze_crypto_category(
                category='defi',
                timeframe='1d',
                limit=2
            )
            
            print(f"\nРЕЗУЛЬТАТЫ АНАЛИЗА DeFi ТОКЕНОВ:")
            print("-" * 40)
            
            for pair, result in defi_results['results'].items():
                print(f"\n{pair}:")
                if 'error' in result:
                    print(f"  Ошибка: {result['error']}")
                else:
                    current_price = result.get('current_price', 0)
                    predicted_price = result.get('predicted_price', 0)
                    
                    if current_price > 0 and predicted_price > 0:
                        change_percent = ((predicted_price - current_price) / current_price) * 100
                        print(f"  Текущая цена: ${current_price:,.2f}")
                        print(f"  Предсказанная цена: ${predicted_price:,.2f}")
                        print(f"  Изменение: {change_percent:+.2f}%")
            
            # Выводим сводку по DeFi
            defi_summary = defi_results.get('summary', {})
            if defi_summary and 'message' not in defi_summary:
                print(f"\nСВОДКА ПО DeFi:")
                print(f"  Общий тренд: {defi_summary['category_trend']}")
                print(f"  Среднее изменение: {defi_summary['average_price_change']:+.2f}%")
            
            print(f"\nАНАЛИЗ DeFi ТОКЕНОВ ЗАВЕРШЕН!")
            
        except Exception as e:
            print(f"Ошибка при анализе DeFi токенов: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "="*60)
        print(f"ТЕСТ МНОЖЕСТВЕННЫХ КРИПТОВАЛЮТ ЗАВЕРШЕН!")
        print(f"Система поддерживает {len(supported_pairs)} криптовалютных пар")
        print(f"в {len(categories)} категориях для комплексного анализа")

if __name__ == "__main__":
    test_multiple_cryptos()
