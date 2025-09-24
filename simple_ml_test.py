#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ML System Test with Enhanced Logging
Простое тестирование ML системы с детальным логированием
"""

import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import json

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('ML_TEST')

# Подавляем предупреждения
warnings.filterwarnings('ignore')

class SimpleMLTester:
    """
    Простой тестер ML системы с логированием
    """
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.logger = logger
        
        # Создаем директории для результатов
        os.makedirs('test_results', exist_ok=True)
        
        self.logger.info("=== ML SYSTEM TESTER STARTED ===")
        self.logger.info(f"Start time: {datetime.now()}")
    
    def log_step(self, step_name: str, status: str = "START", details: str = ""):
        """Логирование шагов с временными метками"""
        elapsed = time.time() - self.start_time
        self.logger.info(f"[{elapsed:.2f}s] {step_name} - {status} {details}")
    
    def test_dependencies(self) -> bool:
        """Тестирование зависимостей"""
        self.log_step("Testing Dependencies", "START")
        
        try:
            # Основные зависимости
            import flask
            import numpy
            import pandas
            import sklearn
            import talib
            
            self.log_step("Core Libraries", "SUCCESS", f"Flask: {flask.__version__}, NumPy: {numpy.__version__}")
            
            # ML библиотеки
            optional_libs = {}
            try:
                import torch
                optional_libs['PyTorch'] = torch.__version__
                self.log_step("PyTorch", "SUCCESS", f"Version: {torch.__version__}")
            except ImportError:
                self.log_step("PyTorch", "WARNING", "Not installed")
            
            try:
                import xgboost
                optional_libs['XGBoost'] = xgboost.__version__
                self.log_step("XGBoost", "SUCCESS", f"Version: {xgboost.__version__}")
            except ImportError:
                self.log_step("XGBoost", "WARNING", "Not installed")
            
            self.test_results['dependencies'] = {
                'status': 'SUCCESS',
                'optional_libraries': optional_libs
            }
            
            return True
            
        except Exception as e:
            self.log_step("Dependencies", "ERROR", str(e))
            self.test_results['dependencies'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_feature_engineering(self) -> bool:
        """Тестирование создания признаков"""
        self.log_step("Testing Feature Engineering", "START")
        
        try:
            from app.services.analysis_service import AdvancedFeatureEngineering
            
            # Создаем тестовые данные
            np.random.seed(42)
            n_samples = 500
            
            # Генерируем реалистичные OHLCV данные
            base_price = 50000
            price_changes = np.random.normal(0, 0.02, n_samples)
            prices = [base_price]
            
            for change in price_changes[1:]:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1000))
            
            # Создаем OHLCV данные
            test_data = []
            for i, price in enumerate(prices):
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.uniform(1000, 10000)
                
                test_data.append({
                    'open': prices[i-1] if i > 0 else price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': volume
                })
            
            df_test = pd.DataFrame(test_data)
            self.log_step("Test Data Created", "SUCCESS", f"Created {len(df_test)} records")
            
            # Тестируем создание признаков
            feature_engineer = AdvancedFeatureEngineering()
            df_features = feature_engineer.create_advanced_features(df_test.copy())
            
            feature_count = len(df_features.columns)
            self.log_step("Feature Creation", "SUCCESS", f"Created {feature_count} features")
            
            # Анализируем качество признаков
            valid_features = 0
            nan_features = 0
            constant_features = 0
            
            for col in df_features.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    if df_features[col].isna().all():
                        nan_features += 1
                    elif df_features[col].std() < 1e-8:
                        constant_features += 1
                    else:
                        valid_features += 1
            
            self.log_step("Feature Quality Analysis", "SUCCESS", 
                         f"Valid: {valid_features}, NaN: {nan_features}, Constant: {constant_features}")
            
            self.test_results['feature_engineering'] = {
                'status': 'SUCCESS',
                'total_features': feature_count,
                'valid_features': valid_features,
                'nan_features': nan_features,
                'constant_features': constant_features
            }
            
            return True
            
        except Exception as e:
            self.log_step("Feature Engineering", "ERROR", str(e))
            self.test_results['feature_engineering'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def test_ensemble_model(self) -> bool:
        """Тестирование Ensemble модели"""
        self.log_step("Testing Ensemble Model", "START")
        
        try:
            from app.services.analysis_service import EnsembleMLModel, AdvancedFeatureEngineering
            
            # Создаем тестовые данные для обучения
            np.random.seed(42)
            n_samples = 1000
            
            # Генерируем синтетические данные с трендом
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
            self.log_step("Synthetic Data", "SUCCESS", f"Created {len(df_test)} samples")
            
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
            
            self.log_step("Data Preparation", "SUCCESS", f"X: {X.shape}, y: {y.shape}")
            
            # Создаем и обучаем модель
            ensemble_model = EnsembleMLModel()
            
            self.log_step("Model Training", "START", "Starting ensemble model training...")
            training_start = time.time()
            
            # Создаем Flask app context для тестирования
            from app import create_app
            app = create_app()
            
            with app.app_context():
                training_metrics = ensemble_model.train(X, y)
            
            training_time = time.time() - training_start
            self.log_step("Model Training", "SUCCESS", f"Completed in {training_time:.2f}s")
            
            # Логируем метрики каждой модели
            for model_name, metrics in training_metrics.items():
                if isinstance(metrics, dict):
                    accuracy = metrics.get('accuracy', 0)
                    r2 = metrics.get('r2', 0)
                    mae = metrics.get('mae', 0)
                    self.log_step(f"Model {model_name}", "SUCCESS", 
                                f"Accuracy: {accuracy:.2f}%, R2: {r2:.3f}, MAE: {mae:.2f}")
            
            # Тестируем предсказание
            test_sample = X.iloc[-5:].copy()
            predictions = []
            
            with app.app_context():
                for i in range(len(test_sample)):
                    pred_price, uncertainty, details = ensemble_model.predict(test_sample.iloc[i:i+1])
                    predictions.append({
                        'predicted_price': pred_price,
                        'uncertainty': uncertainty,
                        'actual_price': y.iloc[-(5-i)],
                        'models_used': details['models_used']
                    })
                    
                    self.log_step(f"Prediction {i+1}", "SUCCESS", 
                                f"Price: {pred_price:.2f}, Uncertainty: +/-{uncertainty:.2f}")
            
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
            self.log_step("Ensemble Model", "ERROR", str(e))
            self.test_results['ensemble_model'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def run_comprehensive_test(self):
        """Запускает полный комплексный тест ML системы"""
        self.log_step("COMPREHENSIVE ML SYSTEM TEST", "START")
        
        # Тест 1: Зависимости
        self.test_dependencies()
        
        # Тест 2: Feature Engineering
        self.test_feature_engineering()
        
        # Тест 3: Ensemble модель
        self.test_ensemble_model()
        
        # Финальный отчет
        self.generate_final_report()
        
        total_time = time.time() - self.start_time
        self.log_step("TESTING COMPLETED", "SUCCESS", f"Total time: {total_time:.2f}s")
    
    def generate_final_report(self):
        """Генерирует финальный отчет о тестировании"""
        self.log_step("Generating Final Report", "START")
        
        total_time = time.time() - self.start_time
        
        # Создаем текстовый отчет
        report = f"""
{'='*80}
ML SYSTEM TEST REPORT
{'='*80}
Test Duration: {total_time:.2f} seconds
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TEST RESULTS:
"""
        
        # Добавляем результаты каждого теста
        for test_name, result in self.test_results.items():
            status_icon = 'PASS' if result.get('status') == 'SUCCESS' else 'FAIL'
            report += f"\n[{status_icon}] {test_name.replace('_', ' ').title()}: {result.get('status', 'UNKNOWN')}\n"
            
            if result.get('status') == 'SUCCESS':
                if test_name == 'ensemble_model':
                    report += f"   - Training Time: {result.get('training_time', 0):.2f} sec\n"
                    report += f"   - Best Accuracy: {result.get('best_accuracy', 0):.2f}%\n"
                    report += f"   - Models Trained: {result.get('models_trained', 0)}\n"
                elif test_name == 'feature_engineering':
                    report += f"   - Total Features: {result.get('total_features', 0)}\n"
                    report += f"   - Valid Features: {result.get('valid_features', 0)}\n"
            else:
                report += f"   - Error: {result.get('error', 'Unknown error')}\n"
        
        # Общая статистика
        successful_tests = sum(1 for r in self.test_results.values() if r.get('status') == 'SUCCESS')
        total_tests = len(self.test_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report += f"""
{'='*80}
SUMMARY:
- Total Tests: {total_tests}
- Successful Tests: {successful_tests}
- Success Rate: {success_rate:.1f}%
- Total Time: {total_time:.2f} seconds
{'='*80}
"""
        
        # Сохраняем отчет
        with open('test_results/ml_system_test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Сохраняем JSON результаты
        with open('test_results/ml_system_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.log_step("Report Saved", "SUCCESS", "test_results/ml_system_test_report.txt")
        
        # Выводим краткую сводку
        print("\n" + "="*80)
        print("ML SYSTEM TEST SUMMARY:")
        print("="*80)
        print(f"Successful tests: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Reports saved in: test_results/")
        print("="*80)

def main():
    """Главная функция для запуска тестирования"""
    print("Starting comprehensive ML system testing...")
    print("="*80)
    
    tester = SimpleMLTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
