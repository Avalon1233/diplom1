# app/services/backtesting_service.py
"""
Комплексная система бэктестирования и валидации ML-моделей для криптовалютного анализа
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os
from flask import current_app

from app.services.analysis_service import AnalysisService, EnsembleMLModel, AdvancedFeatureEngineering
from app.services.crypto_service import CryptoService

warnings.filterwarnings('ignore')


class BacktestingService:
    """
    Сервис для комплексного бэктестирования ML-моделей
    """
    
    def __init__(self):
        self.crypto_service = CryptoService()
        self.analysis_service = AnalysisService()
        self.feature_engineer = AdvancedFeatureEngineering()
        
    def comprehensive_backtest(self, symbol: str, timeframe: str, 
                             test_periods: int = 30, 
                             retrain_frequency: int = 7) -> Dict[str, Any]:
        """
        Проводит комплексное бэктестирование модели
        
        Args:
            symbol: Торговая пара
            timeframe: Временной интервал
            test_periods: Количество периодов для тестирования
            retrain_frequency: Частота переобучения модели (в днях)
            
        Returns:
            Детальный отчет о производительности модели
        """
        current_app.logger.info(f"🔬 Начинаем комплексное бэктестирование для {symbol} ({timeframe})")
        
        # Получаем расширенные исторические данные
        df = self._get_extended_historical_data(symbol, timeframe, periods=365)
        
        if len(df) < 100:
            raise ValueError(f"Недостаточно данных для бэктестирования: {len(df)} записей")
            
        # Создаем признаки
        df_features = self.feature_engineer.create_advanced_features(df)
        
        # Подготавливаем данные для тестирования
        features = [col for col in df_features.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']]
        
        df_features['target'] = df_features['close'].shift(-1)
        df_features = df_features.dropna(subset=['target'])
        
        X = df_features[features]
        y = df_features['target']
        
        # Проводим walk-forward анализ
        results = self._walk_forward_analysis(X, y, test_periods, retrain_frequency)
        
        # Анализируем результаты
        performance_metrics = self._calculate_performance_metrics(results)
        
        # Создаем визуализации
        charts = self._create_performance_charts(results, symbol)
        
        # Анализ важности признаков
        feature_importance = self._analyze_feature_importance(X, y)
        
        current_app.logger.info(f"✅ Бэктестирование завершено. Точность: {performance_metrics['accuracy']:.2f}%")
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'test_periods': test_periods,
            'performance_metrics': performance_metrics,
            'predictions_vs_actual': results,
            'feature_importance': feature_importance,
            'charts': charts,
            'recommendations': self._generate_model_recommendations(performance_metrics)
        }
    
    def _get_extended_historical_data(self, symbol: str, timeframe: str, periods: int = 365) -> pd.DataFrame:
        """Получает расширенные исторические данные"""
        try:
            # Используем существующий метод из analysis_service
            return self.analysis_service._get_historical_data(symbol, timeframe, extended=True)
        except Exception as e:
            current_app.logger.error(f"Ошибка получения данных: {e}")
            # Создаем синтетические данные для тестирования
            return self._generate_synthetic_data(periods)
    
    def _generate_synthetic_data(self, periods: int) -> pd.DataFrame:
        """Генерирует синтетические данные для тестирования"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=periods), 
                             end=datetime.now(), freq='D')
        
        # Генерируем реалистичные ценовые данные с трендом и волатильностью
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0.001, 0.02, len(dates))  # Средняя доходность 0.1% с волатильностью 2%
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Создаем OHLCV данные
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, len(dates))
        })
        
        # Корректируем high/low
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        return df
    
    def _walk_forward_analysis(self, X: pd.DataFrame, y: pd.Series, 
                              test_periods: int, retrain_frequency: int) -> List[Dict]:
        """
        Проводит walk-forward анализ модели
        """
        results = []
        total_samples = len(X)
        
        # Начинаем с 70% данных для первоначального обучения
        initial_train_size = int(total_samples * 0.7)
        
        current_app.logger.info(f"📊 Walk-forward анализ: {test_periods} периодов, переобучение каждые {retrain_frequency} дней")
        
        model = None
        days_since_retrain = 0
        
        for i in range(initial_train_size, min(total_samples - 1, initial_train_size + test_periods)):
            # Переобучаем модель при необходимости
            if model is None or days_since_retrain >= retrain_frequency:
                current_app.logger.info(f"🔄 Переобучение модели на периоде {i}")
                
                # Используем скользящее окно для обучения
                train_start = max(0, i - 200)  # Используем последние 200 точек для обучения
                X_train = X.iloc[train_start:i]
                y_train = y.iloc[train_start:i]
                
                model = EnsembleMLModel()
                try:
                    training_metrics = model.train(X_train, y_train)
                    days_since_retrain = 0
                except Exception as e:
                    current_app.logger.error(f"Ошибка обучения модели: {e}")
                    continue
            
            # Делаем предсказание
            try:
                X_test = X.iloc[i:i+1]
                actual_price = y.iloc[i]
                
                predicted_price, std_dev, details = model.predict(X_test)
                
                # Рассчитываем метрики
                error = abs(predicted_price - actual_price)
                percentage_error = (error / actual_price) * 100
                
                direction_actual = 1 if i > 0 and actual_price > y.iloc[i-1] else 0
                direction_predicted = 1 if i > 0 and predicted_price > y.iloc[i-1] else 0
                direction_correct = direction_actual == direction_predicted
                
                results.append({
                    'period': i,
                    'actual_price': actual_price,
                    'predicted_price': predicted_price,
                    'error': error,
                    'percentage_error': percentage_error,
                    'direction_correct': direction_correct,
                    'std_dev': std_dev,
                    'confidence': 1 / (1 + percentage_error)  # Простая мера уверенности
                })
                
                days_since_retrain += 1
                
            except Exception as e:
                current_app.logger.error(f"Ошибка предсказания на периоде {i}: {e}")
                continue
        
        return results
    
    def _calculate_performance_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Рассчитывает метрики производительности модели"""
        if not results:
            return {}
        
        df_results = pd.DataFrame(results)
        
        # Основные метрики точности
        mae = df_results['error'].mean()
        rmse = np.sqrt(df_results['error'].pow(2).mean())
        mape = df_results['percentage_error'].mean()
        
        # Точность направления движения
        direction_accuracy = df_results['direction_correct'].mean() * 100
        
        # R² score
        actual_prices = df_results['actual_price'].values
        predicted_prices = df_results['predicted_price'].values
        r2 = r2_score(actual_prices, predicted_prices)
        
        # Финансовые метрики
        returns_actual = np.diff(actual_prices) / actual_prices[:-1]
        returns_predicted = np.diff(predicted_prices) / actual_prices[:-1]
        
        # Sharpe ratio (упрощенный)
        sharpe_ratio = np.mean(returns_predicted) / np.std(returns_predicted) if np.std(returns_predicted) > 0 else 0
        
        # Максимальная просадка
        cumulative_returns = np.cumprod(1 + returns_predicted)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Общая точность (комбинированная метрика)
        accuracy = max(0, min(100, (1 - mape/100) * 100))
        
        return {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'r2_score': r2,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_predictions': len(results),
            'confidence_avg': df_results['confidence'].mean()
        }
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Анализирует важность признаков"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Обучаем простую модель для анализа важности признаков
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X.fillna(0), y)
            
            importance_dict = dict(zip(X.columns, rf.feature_importances_))
            
            # Сортируем по важности
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            current_app.logger.error(f"Ошибка анализа важности признаков: {e}")
            return {}
    
    def _create_performance_charts(self, results: List[Dict], symbol: str) -> Dict[str, str]:
        """Создает графики производительности модели"""
        try:
            df_results = pd.DataFrame(results)
            
            # График предсказаний vs реальных значений
            plt.figure(figsize=(15, 10))
            
            # Subplot 1: Предсказания vs Реальные значения
            plt.subplot(2, 2, 1)
            plt.plot(df_results['actual_price'], label='Реальная цена', alpha=0.7)
            plt.plot(df_results['predicted_price'], label='Предсказанная цена', alpha=0.7)
            plt.title(f'Предсказания vs Реальные значения - {symbol}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Ошибки предсказаний
            plt.subplot(2, 2, 2)
            plt.plot(df_results['percentage_error'])
            plt.title('Процентная ошибка предсказаний')
            plt.ylabel('Ошибка (%)')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Точность направления
            plt.subplot(2, 2, 3)
            rolling_accuracy = df_results['direction_correct'].rolling(window=10).mean() * 100
            plt.plot(rolling_accuracy)
            plt.title('Точность направления (скользящее среднее 10 периодов)')
            plt.ylabel('Точность (%)')
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: Распределение ошибок
            plt.subplot(2, 2, 4)
            plt.hist(df_results['percentage_error'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Распределение процентных ошибок')
            plt.xlabel('Ошибка (%)')
            plt.ylabel('Частота')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохраняем график
            chart_path = f'static/charts/backtest_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {'performance_chart': chart_path}
            
        except Exception as e:
            current_app.logger.error(f"Ошибка создания графиков: {e}")
            return {}
    
    def _generate_model_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Генерирует рекомендации по улучшению модели"""
        recommendations = []
        
        accuracy = metrics.get('accuracy', 0)
        direction_accuracy = metrics.get('direction_accuracy', 0)
        r2_score = metrics.get('r2_score', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        if accuracy < 85:
            recommendations.append("🔧 Низкая общая точность. Рекомендуется добавить больше признаков или улучшить предобработку данных.")
        
        if direction_accuracy < 60:
            recommendations.append("📈 Низкая точность направления движения. Рассмотрите добавление momentum индикаторов.")
        
        if r2_score < 0.3:
            recommendations.append("📊 Низкий R² score. Модель плохо объясняет вариацию данных. Попробуйте другие алгоритмы.")
        
        if sharpe_ratio < 0.5:
            recommendations.append("💰 Низкий Sharpe ratio. Стратегия может быть нерентабельной с учетом риска.")
        
        if accuracy > 90 and direction_accuracy > 70:
            recommendations.append("✅ Отличная производительность модели! Можно использовать в продакшене.")
        
        if not recommendations:
            recommendations.append("📋 Модель показывает хорошие результаты. Продолжайте мониторинг производительности.")
        
        return recommendations
    
    def cross_validation_analysis(self, symbol: str, timeframe: str, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Проводит cross-validation анализ модели
        """
        current_app.logger.info(f"🔄 Cross-validation анализ для {symbol} с {cv_folds} фолдами")
        
        # Получаем данные
        df = self._get_extended_historical_data(symbol, timeframe)
        df_features = self.feature_engineer.create_advanced_features(df)
        
        features = [col for col in df_features.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']]
        
        df_features['target'] = df_features['close'].shift(-1)
        df_features = df_features.dropna(subset=['target'])
        
        X = df_features[features].fillna(0)
        y = df_features['target']
        
        # Time Series Split для временных рядов
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            current_app.logger.info(f"📊 Обработка фолда {fold + 1}/{cv_folds}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Обучаем модель
            model = EnsembleMLModel()
            try:
                training_metrics = model.train(X_train, y_train)
                
                # Тестируем на тестовой выборке
                fold_results = []
                for i in range(len(X_test)):
                    try:
                        pred_price, std_dev, details = model.predict(X_test.iloc[i:i+1])
                        actual_price = y_test.iloc[i]
                        
                        error = abs(pred_price - actual_price)
                        percentage_error = (error / actual_price) * 100
                        
                        fold_results.append({
                            'predicted': pred_price,
                            'actual': actual_price,
                            'error': error,
                            'percentage_error': percentage_error
                        })
                    except Exception as e:
                        continue
                
                if fold_results:
                    fold_df = pd.DataFrame(fold_results)
                    fold_metrics = {
                        'fold': fold + 1,
                        'mae': fold_df['error'].mean(),
                        'mape': fold_df['percentage_error'].mean(),
                        'r2': r2_score(fold_df['actual'], fold_df['predicted']),
                        'samples': len(fold_results)
                    }
                    cv_results.append(fold_metrics)
                    
            except Exception as e:
                current_app.logger.error(f"Ошибка в фолде {fold + 1}: {e}")
                continue
        
        # Агрегируем результаты
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            aggregated_metrics = {
                'mean_mae': cv_df['mae'].mean(),
                'std_mae': cv_df['mae'].std(),
                'mean_mape': cv_df['mape'].mean(),
                'std_mape': cv_df['mape'].std(),
                'mean_r2': cv_df['r2'].mean(),
                'std_r2': cv_df['r2'].std(),
                'cv_folds': cv_folds,
                'fold_results': cv_results
            }
            
            current_app.logger.info(f"✅ Cross-validation завершен. Средняя MAPE: {aggregated_metrics['mean_mape']:.2f}%")
            return aggregated_metrics
        else:
            return {'error': 'Не удалось выполнить cross-validation'}


def run_comprehensive_model_evaluation(symbol: str = 'BTC-USD', timeframe: str = '1d') -> Dict[str, Any]:
    """
    Запускает полную оценку модели с бэктестированием и cross-validation
    """
    backtesting_service = BacktestingService()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe
    }
    
    try:
        # Бэктестирование
        current_app.logger.info("🚀 Запуск комплексного бэктестирования...")
        backtest_results = backtesting_service.comprehensive_backtest(
            symbol=symbol, 
            timeframe=timeframe,
            test_periods=30,
            retrain_frequency=7
        )
        results['backtest'] = backtest_results
        
        # Cross-validation
        current_app.logger.info("🔄 Запуск cross-validation анализа...")
        cv_results = backtesting_service.cross_validation_analysis(
            symbol=symbol,
            timeframe=timeframe,
            cv_folds=5
        )
        results['cross_validation'] = cv_results
        
        # Общие рекомендации
        results['overall_recommendations'] = _generate_overall_recommendations(
            backtest_results.get('performance_metrics', {}),
            cv_results
        )
        
        current_app.logger.info("✅ Полная оценка модели завершена успешно")
        
    except Exception as e:
        current_app.logger.error(f"❌ Ошибка при оценке модели: {e}")
        results['error'] = str(e)
    
    return results


def _generate_overall_recommendations(backtest_metrics: Dict, cv_metrics: Dict) -> List[str]:
    """Генерирует общие рекомендации на основе всех тестов"""
    recommendations = []
    
    bt_accuracy = backtest_metrics.get('accuracy', 0)
    cv_mape = cv_metrics.get('mean_mape', 100)
    cv_std = cv_metrics.get('std_mape', 0)
    
    if bt_accuracy > 85 and cv_mape < 15:
        recommendations.append("🎯 ОТЛИЧНАЯ МОДЕЛЬ: Высокая точность на бэктесте и стабильность на cross-validation")
    elif bt_accuracy > 75 and cv_mape < 25:
        recommendations.append("✅ ХОРОШАЯ МОДЕЛЬ: Приемлемая точность, можно использовать с осторожностью")
    else:
        recommendations.append("⚠️ ТРЕБУЕТ УЛУЧШЕНИЯ: Низкая точность, необходима доработка")
    
    if cv_std > 10:
        recommendations.append("📊 Высокая вариативность результатов. Рекомендуется стабилизация модели")
    
    return recommendations
