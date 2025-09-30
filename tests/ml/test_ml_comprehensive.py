#!/usr/bin/env python3
"""
Комплексное тестирование ML-модели с бэктестированием и анализом точности
"""
import sys
import os
import json
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services.backtesting_service import run_comprehensive_model_evaluation, BacktestingService


def main():
    """Основная функция для запуска комплексного тестирования"""
    print("Запуск комплексного тестирования ML-модели")
    print("=" * 60)
    
    # Создаем контекст приложения
    app = create_app()
    
    with app.app_context():
        try:
            # Тестируем разные криптовалюты и временные интервалы
            test_configs = [
                {'symbol': 'BTC-USD', 'timeframe': '1d'},
                {'symbol': 'ETH-USD', 'timeframe': '1d'},
                {'symbol': 'BTC-USD', 'timeframe': '1w'},
            ]
            
            all_results = {}
            
            for config in test_configs:
                symbol = config['symbol']
                timeframe = config['timeframe']
                
                print(f"\n📊 Тестирование {symbol} ({timeframe})")
                print("-" * 40)
                
                try:
                    # Запускаем полную оценку модели
                    results = run_comprehensive_model_evaluation(symbol, timeframe)
                    all_results[f"{symbol}_{timeframe}"] = results
                    
                    # Выводим основные метрики
                    if 'backtest' in results and 'performance_metrics' in results['backtest']:
                        metrics = results['backtest']['performance_metrics']
                        print(f"✅ Точность: {metrics.get('accuracy', 0):.2f}%")
                        print(f"📈 Точность направления: {metrics.get('direction_accuracy', 0):.2f}%")
                        print(f"📊 R² Score: {metrics.get('r2_score', 0):.3f}")
                        print(f"💰 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                        print(f"📉 Макс. просадка: {metrics.get('max_drawdown', 0):.2f}%")
                    
                    if 'cross_validation' in results:
                        cv_metrics = results['cross_validation']
                        print(f"🔄 CV MAPE: {cv_metrics.get('mean_mape', 0):.2f}% ± {cv_metrics.get('std_mape', 0):.2f}%")
                    
                    # Выводим рекомендации
                    if 'overall_recommendations' in results:
                        print("\n💡 Рекомендации:")
                        for rec in results['overall_recommendations']:
                            print(f"   {rec}")
                    
                except Exception as e:
                    print(f"❌ Ошибка при тестировании {symbol} ({timeframe}): {e}")
                    all_results[f"{symbol}_{timeframe}"] = {'error': str(e)}
            
            # Сохраняем результаты в файл
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"ml_comprehensive_test_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Результаты сохранены в {results_file}")
            
            # Создаем сводный отчет
            create_summary_report(all_results, timestamp)
            
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            return 1
    
    print("\n✅ Комплексное тестирование завершено!")
    return 0


def create_summary_report(results: dict, timestamp: str):
    """Создает сводный отчет по результатам тестирования"""
    report_file = f"ml_test_summary_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 📊 Сводный отчет по тестированию ML-модели\n\n")
        f.write(f"**Дата тестирования:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Общая статистика
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if 'error' not in r)
        
        f.write(f"**Общая статистика:**\n")
        f.write(f"- Всего тестов: {total_tests}\n")
        f.write(f"- Успешных: {successful_tests}\n")
        f.write(f"- Неудачных: {total_tests - successful_tests}\n\n")
        
        # Детальные результаты
        f.write("## 📈 Детальные результаты\n\n")
        
        for test_name, result in results.items():
            f.write(f"### {test_name}\n\n")
            
            if 'error' in result:
                f.write(f"❌ **Ошибка:** {result['error']}\n\n")
                continue
            
            # Бэктест результаты
            if 'backtest' in result and 'performance_metrics' in result['backtest']:
                metrics = result['backtest']['performance_metrics']
                f.write("**Бэктест метрики:**\n")
                f.write(f"- Точность: {metrics.get('accuracy', 0):.2f}%\n")
                f.write(f"- Точность направления: {metrics.get('direction_accuracy', 0):.2f}%\n")
                f.write(f"- R² Score: {metrics.get('r2_score', 0):.3f}\n")
                f.write(f"- MAE: {metrics.get('mae', 0):.2f}\n")
                f.write(f"- MAPE: {metrics.get('mape', 0):.2f}%\n")
                f.write(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"- Макс. просадка: {metrics.get('max_drawdown', 0):.2f}%\n\n")
            
            # Cross-validation результаты
            if 'cross_validation' in result:
                cv = result['cross_validation']
                f.write("**Cross-Validation метрики:**\n")
                f.write(f"- Средняя MAPE: {cv.get('mean_mape', 0):.2f}% ± {cv.get('std_mape', 0):.2f}%\n")
                f.write(f"- Средний R²: {cv.get('mean_r2', 0):.3f} ± {cv.get('std_r2', 0):.3f}\n")
                f.write(f"- Количество фолдов: {cv.get('cv_folds', 0)}\n\n")
            
            # Рекомендации
            if 'overall_recommendations' in result:
                f.write("**Рекомендации:**\n")
                for rec in result['overall_recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
        
        # Общие выводы
        f.write("## 🎯 Общие выводы\n\n")
        
        # Рассчитываем средние метрики
        accuracies = []
        direction_accuracies = []
        r2_scores = []
        
        for result in results.values():
            if 'error' not in result and 'backtest' in result:
                metrics = result['backtest'].get('performance_metrics', {})
                if metrics.get('accuracy'):
                    accuracies.append(metrics['accuracy'])
                if metrics.get('direction_accuracy'):
                    direction_accuracies.append(metrics['direction_accuracy'])
                if metrics.get('r2_score'):
                    r2_scores.append(metrics['r2_score'])
        
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            f.write(f"- **Средняя точность:** {avg_accuracy:.2f}%\n")
            
            if avg_accuracy >= 85:
                f.write("- ✅ **Оценка:** ОТЛИЧНАЯ модель, готова к продакшену\n")
            elif avg_accuracy >= 75:
                f.write("- ⚠️ **Оценка:** ХОРОШАЯ модель, требует мониторинга\n")
            else:
                f.write("- ❌ **Оценка:** Модель требует значительных улучшений\n")
        
        if direction_accuracies:
            avg_direction = sum(direction_accuracies) / len(direction_accuracies)
            f.write(f"- **Средняя точность направления:** {avg_direction:.2f}%\n")
        
        if r2_scores:
            avg_r2 = sum(r2_scores) / len(r2_scores)
            f.write(f"- **Средний R² Score:** {avg_r2:.3f}\n")
    
    print(f"📄 Сводный отчет сохранен в {report_file}")


def run_quick_test():
    """Быстрый тест одной конфигурации"""
    print("Запуск быстрого теста модели")
    
    app = create_app()
    with app.app_context():
        backtesting_service = BacktestingService()
        
        try:
            # Быстрый бэктест на 10 периодах
            results = backtesting_service.comprehensive_backtest(
                symbol='BTC-USD',
                timeframe='1d',
                test_periods=10,
                retrain_frequency=5
            )
            
            metrics = results['performance_metrics']
            print(f"✅ Быстрый тест завершен:")
            print(f"   Точность: {metrics.get('accuracy', 0):.2f}%")
            print(f"   Точность направления: {metrics.get('direction_accuracy', 0):.2f}%")
            print(f"   R² Score: {metrics.get('r2_score', 0):.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка быстрого теста: {e}")
            return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Комплексное тестирование ML-модели')
    parser.add_argument('--quick', action='store_true', help='Запустить быстрый тест')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        exit_code = main()
        sys.exit(exit_code)
