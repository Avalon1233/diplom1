#!/usr/bin/env python3
"""
Быстрое тестирование ML-модели
"""
import sys
import os
import traceback
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.analysis_service import AnalysisService


def test_model_accuracy():
    """Тестирует точность ML-модели"""
    print("Запуск быстрого теста ML-модели")
    print("-" * 50)
    
    app = create_app()
    
    with app.app_context():
        try:
            analysis_service = AnalysisService()
            
            # Тестируем анализ для BTC
            print("Тестирование анализа BTC-USD...")
            result = analysis_service.advanced_ml_analysis('BTC-USD', '1d')
            
            if result.get('status') == 'success':
                print("УСПЕХ: Анализ выполнен успешно")
                print(f"Текущая цена: ${result.get('current_price', 0):,.2f}")
                print(f"Прогнозная цена: ${result.get('predicted_price', 0):,.2f}")
                print(f"Стандартное отклонение: {result.get('prediction_std_dev', 0):.2f}")
                try:
                    print(f"Рекомендация: {result.get('recommendation', 'Нет')}")
                except UnicodeEncodeError:
                    print("Рекомендация: [Проблема с кодировкой]")
                
                # Проверяем метрики обучения
                training_metrics = result.get('training_metrics', {})
                if training_metrics:
                    print("\nМетрики обучения:")
                    for model_name, metrics in training_metrics.items():
                        if isinstance(metrics, dict) and 'accuracy' in metrics:
                            print(f"  {model_name}: Точность {metrics['accuracy']:.2f}%, R2 {metrics.get('r2', 0):.3f}")
                
                # Проверяем детали предсказания
                prediction_details = result.get('prediction_details', {})
                if prediction_details:
                    individual_preds = prediction_details.get('individual_predictions', {})
                    print(f"\nИндивидуальные предсказания:")
                    for model, pred in individual_preds.items():
                        print(f"  {model}: ${pred:,.2f}")
                
                return True
            else:
                print(f"ОШИБКА: {result.get('message', 'Неизвестная ошибка')}")
                return False
                
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
            print("Детали ошибки:")
            traceback.print_exc()
            return False


def test_multiple_symbols():
    """Тестирует модель на нескольких символах"""
    print("\nТестирование нескольких криптовалют")
    print("-" * 50)
    
    symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD']
    results = {}
    
    app = create_app()
    
    with app.app_context():
        analysis_service = AnalysisService()
        
        for symbol in symbols:
            print(f"\nТестирование {symbol}...")
            try:
                result = analysis_service.advanced_ml_analysis(symbol, '1d')
                
                if result.get('status') == 'success':
                    results[symbol] = {
                        'success': True,
                        'current_price': result.get('current_price', 0),
                        'predicted_price': result.get('predicted_price', 0),
                        'accuracy': 0
                    }
                    
                    # Извлекаем точность из метрик
                    training_metrics = result.get('training_metrics', {})
                    if 'ensemble' in training_metrics:
                        results[symbol]['accuracy'] = training_metrics['ensemble'].get('accuracy', 0)
                    
                    print(f"  УСПЕХ: Точность {results[symbol]['accuracy']:.2f}%")
                else:
                    results[symbol] = {'success': False, 'error': result.get('message', 'Ошибка')}
                    print(f"  ОШИБКА: {results[symbol]['error']}")
                    
            except Exception as e:
                results[symbol] = {'success': False, 'error': str(e)}
                print(f"  КРИТИЧЕСКАЯ ОШИБКА: {e}")
    
    # Сводка результатов
    print("\n" + "=" * 50)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    successful = 0
    total_accuracy = 0
    
    for symbol, result in results.items():
        if result['success']:
            successful += 1
            accuracy = result.get('accuracy', 0)
            total_accuracy += accuracy
            print(f"{symbol}: УСПЕХ (Точность: {accuracy:.2f}%)")
        else:
            print(f"{symbol}: ОШИБКА - {result['error']}")
    
    if successful > 0:
        avg_accuracy = total_accuracy / successful
        print(f"\nУспешных тестов: {successful}/{len(symbols)}")
        print(f"Средняя точность: {avg_accuracy:.2f}%")
        
        if avg_accuracy >= 90:
            print("ОЦЕНКА: ОТЛИЧНАЯ модель!")
        elif avg_accuracy >= 80:
            print("ОЦЕНКА: ХОРОШАЯ модель")
        elif avg_accuracy >= 70:
            print("ОЦЕНКА: Приемлемая модель")
        else:
            print("ОЦЕНКА: Модель требует улучшения")
    else:
        print("Все тесты завершились неудачно!")
    
    return results


def benchmark_model_speed():
    """Тестирует скорость работы модели"""
    print("\nТестирование скорости модели")
    print("-" * 50)
    
    app = create_app()
    
    with app.app_context():
        analysis_service = AnalysisService()
        
        # Первый запуск (с обучением)
        print("Первый запуск (с обучением модели)...")
        start_time = datetime.now()
        
        try:
            result1 = analysis_service.advanced_ml_analysis('BTC-USD', '1d')
            end_time = datetime.now()
            first_run_time = (end_time - start_time).total_seconds()
            
            print(f"Время первого запуска: {first_run_time:.2f} секунд")
            
            # Второй запуск (с кэшированной моделью)
            print("Второй запуск (с кэшированной моделью)...")
            start_time = datetime.now()
            
            result2 = analysis_service.advanced_ml_analysis('BTC-USD', '1d')
            end_time = datetime.now()
            second_run_time = (end_time - start_time).total_seconds()
            
            print(f"Время второго запуска: {second_run_time:.2f} секунд")
            print(f"Ускорение: {first_run_time/second_run_time:.1f}x")
            
            if second_run_time < 5:
                print("ОТЛИЧНО: Кэширование работает эффективно")
            elif second_run_time < 10:
                print("ХОРОШО: Приемлемая скорость с кэшем")
            else:
                print("ВНИМАНИЕ: Медленная работа даже с кэшем")
                
        except Exception as e:
            print(f"ОШИБКА при тестировании скорости: {e}")


def main():
    """Основная функция"""
    print("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ML-МОДЕЛИ")
    print("=" * 60)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Тест 1: Базовая функциональность
    success1 = test_model_accuracy()
    
    # Тест 2: Множественные символы
    results2 = test_multiple_symbols()
    
    # Тест 3: Скорость работы
    benchmark_model_speed()
    
    # Финальная оценка
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА")
    print("=" * 60)
    
    if success1:
        successful_symbols = sum(1 for r in results2.values() if r.get('success', False))
        total_symbols = len(results2)
        
        print(f"Базовый тест: ПРОЙДЕН")
        print(f"Тесты символов: {successful_symbols}/{total_symbols}")
        
        if successful_symbols == total_symbols:
            print("ОБЩАЯ ОЦЕНКА: ОТЛИЧНО - Модель готова к использованию")
        elif successful_symbols >= total_symbols * 0.7:
            print("ОБЩАЯ ОЦЕНКА: ХОРОШО - Модель работает стабильно")
        else:
            print("ОБЩАЯ ОЦЕНКА: ТРЕБУЕТ УЛУЧШЕНИЯ")
    else:
        print("ОБЩАЯ ОЦЕНКА: КРИТИЧЕСКИЕ ПРОБЛЕМЫ - Модель не работает")
    
    print(f"\nТестирование завершено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
