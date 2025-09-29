"""
Тест оптимизации гиперпараметров для криптовалютных ML моделей
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from app.services.analysis_service import AnalysisService
from app import create_app
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hyperopt_test.log', encoding='utf-8')
    ]
)

def test_hyperparameter_optimization():
    """Тестирование оптимизации гиперпараметров"""
    
    # Создаем Flask приложение для контекста
    app = create_app()
    
    with app.app_context():
        print("ТЕСТ ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ")
        print("=" * 60)
        
        # Инициализируем сервис анализа
        analysis_service = AnalysisService()
        
        # Тестируем оптимизацию для BTC-USD
        symbol = 'BTC-USD'
        print(f"\nНачинаем оптимизацию гиперпараметров для {symbol}")
        
        try:
            # Выполняем оптимизацию с уменьшенной сеткой для быстрого тестирования
            optimization_results = analysis_service.optimize_hyperparameters(
                symbol=symbol,
                timeframe='1d',
                use_reduced_grid=True,  # Быстрая оптимизация
                cv_folds=3
            )
            
            print(f"\nОптимизация для {symbol} завершена успешно!")
            
            # Выводим результаты
            summary = optimization_results.get('optimization_summary', {})
            print(f"\nРЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
            print("-" * 40)
            
            for model_name, results in summary.items():
                print(f"\nМодель: {model_name.upper()}")
                
                best_params = results.get('best_params', {})
                if best_params:
                    print("   Лучшие параметры:")
                    for param, value in best_params.items():
                        print(f"     {param}: {value}")
                
                metrics = results.get('metrics', {})
                if 'error' not in metrics:
                    print("   Метрики производительности:")
                    print(f"     CV MAE: {metrics.get('best_cv_mae', 'N/A'):.2f}")
                    print(f"     Train MAE: {metrics.get('train_mae', 'N/A'):.2f}")
                    print(f"     Train R2: {metrics.get('train_r2', 'N/A'):.3f}")
                    print(f"     Время оптимизации: {metrics.get('optimization_time', 'N/A'):.2f}с")
                else:
                    print(f"   Ошибка: {metrics['error']}")
            
            print(f"\nОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ ЗАВЕРШЕНА УСПЕШНО!")
            print(f"Результаты сохранены в models/advanced_ml/hyperopt_results_{symbol.replace('-', '_')}_1d.joblib")
            
            print(f"\nОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ БУДУТ ИСПОЛЬЗОВАНЫ ПРИ СЛЕДУЮЩЕМ ОБУЧЕНИИ МОДЕЛИ")
            print(f"Лучшие результаты показала модель RF с CV MAE: {summary.get('rf', {}).get('metrics', {}).get('best_cv_mae', 'N/A'):.2f}")
            
        except Exception as e:
            print(f"Ошибка при оптимизации гиперпараметров: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_hyperparameter_optimization()
