#!/usr/bin/env python3
"""
Comprehensive ML System Test with Enhanced Logging and Visualization
Тестирование продвинутой ML системы с детальным логированием и визуализацией
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List
import json

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_system_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('ML_SYSTEM_TEST')

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings('ignore')

# Настройка matplotlib для Windows
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLSystemTester:
    """
    Комплексный тестер ML системы с визуализацией и детальным логированием
    """
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.logger = logger
        
        # Создаем директории для результатов
        os.makedirs('test_results', exist_ok=True)
        os.makedirs('test_results/plots', exist_ok=True)
        os.makedirs('test_results/logs', exist_ok=True)
        
        self.logger.info("🚀 Инициализация ML System Tester")
        self.logger.info("=" * 80)
    
    def log_step(self, step_name: str, status: str = "START", details: str = ""):
        """Логирование шагов с временными метками"""
        elapsed = time.time() - self.start_time
        status_emoji = {
            "START": "🔄",
            "SUCCESS": "✅", 
            "ERROR": "❌",
            "WARNING": "⚠️",
            "INFO": "ℹ️"
        }
        
        emoji = status_emoji.get(status, "📝")
        self.logger.info(f"{emoji} [{elapsed:.2f}s] {step_name} {details}")
    
    def test_imports_and_dependencies(self) -> bool:
        """Тестирование импортов и зависимостей"""
        self.log_step("Тестирование импортов", "START")
        
        try:
            # Основные зависимости
            import flask
            import numpy
            import pandas
            import sklearn
            import talib
            
            self.log_step("Основные библиотеки", "SUCCESS", f"Flask: {flask.__version__}, NumPy: {numpy.__version__}")
            
            # ML библиотеки
            optional_libs = {}
            try:
                import torch
                optional_libs['PyTorch'] = torch.__version__
                self.log_step("PyTorch", "SUCCESS", f"Версия: {torch.__version__}")
            except ImportError:
                self.log_step("PyTorch", "WARNING", "Не установлен")
            
            try:
                import xgboost
                optional_libs['XGBoost'] = xgboost.__version__
                self.log_step("XGBoost", "SUCCESS", f"Версия: {xgboost.__version__}")
            except ImportError:
                self.log_step("XGBoost", "WARNING", "Не установлен")
            
            try:
                import transformers
                optional_libs['Transformers'] = transformers.__version__
                self.log_step("Transformers", "SUCCESS", f"Версия: {transformers.__version__}")
            except ImportError:
                self.log_step("Transformers", "WARNING", "Не установлен")
            
            self.test_results['dependencies'] = {
                'status': 'SUCCESS',
                'optional_libraries': optional_libs
            }
            
            return True
            
        except Exception as e:
            self.log_step("Импорты", "ERROR", str(e))
            self.test_results['dependencies'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_ensemble_model(self) -> bool:
        """Тестирование Ensemble ML модели"""
        self.log_step("Тестирование Ensemble модели", "START")
        
        try:
            from app.services.analysis_service import EnsembleMLModel, AdvancedFeatureEngineering
            
            # Создаем тестовые данные для обучения
            np.random.seed(42)
            n_samples = 1000
            
            # Генерируем синтетические данные с трендом и шумом
            time_series = np.arange(n_samples)
            trend = 0.1 * time_series
            seasonal = 10 * np.sin(2 * np.pi * time_series / 100)
            noise = np.random.normal(0, 5, n_samples)
            prices = 50000 + trend + seasonal + noise
            
            # Создаем DataFrame с OHLCV данными
            test_data = []
            for i in range(n_samples):
                price = prices[i]
                high = price * (1 + abs(np.random.normal(0, 0.005)))
                low = price * (1 - abs(np.random.normal(0, 0.005)))
                volume = np.random.uniform(1000, 5000)
                
                test_data.append({
                    'open': prices[i-1] if i > 0 else price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df_test = pd.DataFrame(test_data)
            self.log_step("Синтетические данные", "SUCCESS", f"Создано {len(df_test)} образцов")
            
            # Создаем признаки
            feature_engineer = AdvancedFeatureEngineering()
            df_features = feature_engineer.create_advanced_features(df_test.copy())
            
            # Подготавливаем данные для обучения
            feature_columns = [col for col in df_features.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            valid_features = []
            for col in feature_columns:
                if not df_features[col].isna().all() and df_features[col].std() > 1e-8:
                    valid_features.append(col)
            
            X = df_features[valid_features].fillna(method='ffill').fillna(method='bfill')
            y = df_features['close'].shift(-1).fillna(method='ffill')
            
            # Удаляем последнюю строку
            X = X.iloc[:-1]
            y = y.iloc[:-1]
            
            self.log_step("Подготовка данных", "SUCCESS", f"X: {X.shape}, y: {y.shape}")
            
            # Создаем и обучаем модель
            ensemble_model = EnsembleMLModel()
            
            self.log_step("Начало обучения модели", "START")
            training_start = time.time()
            
            training_metrics = ensemble_model.train(X, y)
            
            training_time = time.time() - training_start
            self.log_step("Обучение завершено", "SUCCESS", f"Время: {training_time:.2f}с")
            
            # Логируем метрики каждой модели
            for model_name, metrics in training_metrics.items():
                if isinstance(metrics, dict):
                    accuracy = metrics.get('accuracy', 0)
                    r2 = metrics.get('r2', 0)
                    mae = metrics.get('mae', 0)
                    self.log_step(f"Модель {model_name}", "SUCCESS", 
                                f"Точность: {accuracy:.2f}%, R²: {r2:.3f}, MAE: {mae:.2f}")
            
            # Тестируем предсказание
            test_sample = X.iloc[-10:].copy()
            predictions = []
            
            for i in range(len(test_sample)):
                pred_price, uncertainty, details = ensemble_model.predict(test_sample.iloc[i:i+1])
                predictions.append({
                    'predicted_price': pred_price,
                    'uncertainty': uncertainty,
                    'actual_price': y.iloc[-(10-i)],
                    'models_used': details['models_used']
                })
                
                self.log_step(f"Предсказание {i+1}", "SUCCESS", 
                            f"Цена: {pred_price:.2f}, Неопределенность: ±{uncertainty:.2f}")
            
            # Вычисляем общую точность
            best_accuracy = max([m.get('accuracy', 0) for m in training_metrics.values() if isinstance(m, dict)])
            
            self.test_results['ensemble_model'] = {
                'status': 'SUCCESS',
                'training_time': training_time,
                'best_accuracy': best_accuracy,
                'models_trained': len([m for m in training_metrics.keys() if isinstance(training_metrics[m], dict)]),
                'training_metrics': training_metrics,
                'predictions_tested': len(predictions)
            }
            
            return True
            
        except Exception as e:
            self.log_step("Ensemble модель", "ERROR", str(e))
            self.test_results['ensemble_model'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def run_comprehensive_test(self):
        """Запускает полный комплексный тест ML системы"""
        self.log_step("🚀 НАЧАЛО КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ ML СИСТЕМЫ", "START")
        
        # Тест 1: Зависимости
        self.test_imports_and_dependencies()
        
        # Тест 2: Ensemble модель
        self.test_ensemble_model()
        
        # Финальный отчет
        self.generate_final_report()
        
        self.log_step("🎯 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО", "SUCCESS", f"Общее время: {time.time() - self.start_time:.2f}с")
    
    def generate_final_report(self):
        """Генерирует финальный отчет о тестировании"""
        self.log_step("Генерация финального отчета", "START")
        
        total_time = time.time() - self.start_time
        
        # Создаем текстовый отчет
        report = f"""
{'='*80}
🤖 ADVANCED ML SYSTEM TEST REPORT
{'='*80}
Время тестирования: {total_time:.2f} секунд
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

РЕЗУЛЬТАТЫ ТЕСТОВ:
"""
        
        # Добавляем результаты каждого теста
        for test_name, result in self.test_results.items():
            status_icon = '✅' if result.get('status') == 'SUCCESS' else '❌'
            report += f"\n{status_icon} {test_name.replace('_', ' ').title()}: {result.get('status', 'UNKNOWN')}\n"
            
            if result.get('status') == 'SUCCESS':
                if test_name == 'ensemble_model':
                    report += f"   - Время обучения: {result.get('training_time', 0):.2f} сек\n"
                    report += f"   - Лучшая точность: {result.get('best_accuracy', 0):.2f}%\n"
                    report += f"   - Обученных моделей: {result.get('models_trained', 0)}\n"
            else:
                report += f"   - Ошибка: {result.get('error', 'Unknown error')}\n"
        
        # Общая статистика
        successful_tests = sum(1 for r in self.test_results.values() if r.get('status') == 'SUCCESS')
        total_tests = len(self.test_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report += f"""
{'='*80}
ОБЩАЯ СТАТИСТИКА:
- Всего тестов: {total_tests}
- Успешных тестов: {successful_tests}
- Процент успеха: {success_rate:.1f}%
- Общее время: {total_time:.2f} секунд
{'='*80}
"""
        
        # Сохраняем отчет
        with open('test_results/ml_system_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Сохраняем JSON результаты
        with open('test_results/ml_system_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.log_step("Отчет сохранен", "SUCCESS", "test_results/ml_system_test_report.txt")
        
        # Выводим краткую сводку
        print("\n" + "="*80)
        print("🎯 КРАТКАЯ СВОДКА ТЕСТИРОВАНИЯ:")
        print("="*80)
        print(f"✅ Успешных тестов: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"⏱️  Общее время: {total_time:.2f} секунд")
        print(f"📊 Отчеты сохранены в: test_results/")
        print("="*80)

def main():
    """Главная функция для запуска тестирования"""
    print("🚀 Запуск комплексного тестирования ML системы...")
    print("="*80)
    
    tester = MLSystemTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
