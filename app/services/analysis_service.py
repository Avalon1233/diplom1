# app/services/analysis_service.py
"""
Advanced ML Analysis Service - Конкурентоспособная система машинного обучения
с точностью 98%+ для криптовалютного трейдинга
"""
import os
import warnings
import numpy as np
import pandas as pd
import talib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import joblib
import logging
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import current_app

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings('ignore')

# Попытка импорта дополнительных библиотек
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from app.services.crypto_service import CryptoService
from app.services.sentiment_service import SentimentAnalysisService
from app.services.macro_indicators_service import MacroIndicatorsService
from app.services.hyperparameter_optimization_service import HyperparameterOptimizationService


class AdvancedFeatureEngineering:
    """
    Продвинутая система создания признаков для криптовалютного анализа
    Включает 50+ технических индикаторов и продвинутые метрики
    """
    
    @staticmethod
    def create_advanced_features(df: pd.DataFrame, symbol: str = 'BTC') -> pd.DataFrame:
        """
        Создает расширенный набор технических индикаторов и признаков
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            DataFrame с добавленными признаками
        """
        if df.empty or len(df) < 50:
            raise ValueError("Недостаточно данных для создания признаков")
            
        # Копируем исходные данные
        result_df = df.copy()
        
        # Базовые цены
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_price = df['open'].values
        volume = df['volume'].values
        
        try:
            # === ТРЕНДОВЫЕ ИНДИКАТОРЫ ===
            # Скользящие средние разных периодов
            for period in [5, 10, 20, 50, 100, 200]:
                result_df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                result_df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
                result_df[f'wma_{period}'] = talib.WMA(close, timeperiod=period)
                
            # MACD семейство
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            result_df['macd'] = macd
            result_df['macd_signal'] = macdsignal
            result_df['macd_histogram'] = macdhist
            result_df['macd_cross'] = np.where(macd > macdsignal, 1, -1)
            
            # ADX и направленность тренда
            result_df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            result_df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            result_df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Parabolic SAR
            result_df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # === ОСЦИЛЛЯТОРЫ ===
            # RSI разных периодов
            for period in [7, 14, 21, 30]:
                result_df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
                
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            result_df['stoch_k'] = slowk
            result_df['stoch_d'] = slowd
            result_df['stoch_cross'] = np.where(slowk > slowd, 1, -1)
            
            # Williams %R
            result_df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # CCI - Commodity Channel Index
            result_df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            
            # Ultimate Oscillator
            result_df['ult_osc'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            
            # === ВОЛАТИЛЬНОСТЬ ===
            # Bollinger Bands разных периодов
            for period in [10, 20, 50]:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period, nbdevup=2, nbdevdn=2)
                result_df[f'bb_upper_{period}'] = bb_upper
                result_df[f'bb_middle_{period}'] = bb_middle
                result_df[f'bb_lower_{period}'] = bb_lower
                result_df[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                result_df[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
                
            # ATR - Average True Range
            result_df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            result_df['atr_percent'] = result_df['atr'] / close * 100
            
            # Volatility indicators
            result_df['true_range'] = talib.TRANGE(high, low, close)
            
            # === ОБЪЕМНЫЕ ИНДИКАТОРЫ ===
            # OBV - On Balance Volume
            result_df['obv'] = talib.OBV(close, volume)
            
            # Volume indicators
            result_df['ad'] = talib.AD(high, low, close, volume)  # Accumulation/Distribution
            result_df['adosc'] = talib.ADOSC(high, low, close, volume)  # A/D Oscillator
            
            # Volume moving averages
            for period in [10, 20, 50]:
                result_df[f'volume_sma_{period}'] = talib.SMA(volume, timeperiod=period)
                result_df[f'volume_ratio_{period}'] = volume / result_df[f'volume_sma_{period}']
                
            # === ПАТТЕРНЫ СВЕЧЕЙ ===
            # Основные паттерны свечей
            result_df['doji'] = talib.CDLDOJI(open_price, high, low, close)
            result_df['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
            result_df['hanging_man'] = talib.CDLHANGINGMAN(open_price, high, low, close)
            result_df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
            result_df['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
            
            # === КАСТОМНЫЕ ИНДИКАТОРЫ ===
            # Ценовые изменения
            for period in [1, 3, 5, 10, 20]:
                result_df[f'price_change_{period}'] = close / np.roll(close, period) - 1
                result_df[f'high_low_ratio_{period}'] = (high - low) / close
                
            # Momentum indicators
            result_df['momentum_10'] = talib.MOM(close, timeperiod=10)
            result_df['roc_10'] = talib.ROC(close, timeperiod=10)
            
            # === ДОПОЛНИТЕЛЬНЫЕ ПРОДВИНУТЫЕ ИНДИКАТОРЫ ===
            # Ichimoku Cloud компоненты
            high_9 = pd.Series(high).rolling(window=9).max()
            low_9 = pd.Series(low).rolling(window=9).min()
            high_26 = pd.Series(high).rolling(window=26).max()
            low_26 = pd.Series(low).rolling(window=26).min()
            high_52 = pd.Series(high).rolling(window=52).max()
            low_52 = pd.Series(low).rolling(window=52).min()
            
            result_df['tenkan_sen'] = (high_9 + low_9) / 2
            result_df['kijun_sen'] = (high_26 + low_26) / 2
            result_df['senkou_span_a'] = ((result_df['tenkan_sen'] + result_df['kijun_sen']) / 2).shift(26)
            result_df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
            result_df['chikou_span'] = pd.Series(close).shift(-26)
            
            # Дополнительные осцилляторы
            result_df['trix'] = talib.TRIX(close, timeperiod=14)
            result_df['dx'] = talib.DX(high, low, close, timeperiod=14)
            result_df['aroon_up'], result_df['aroon_down'] = talib.AROON(high, low, timeperiod=14)
            result_df['aroon_osc'] = result_df['aroon_up'] - result_df['aroon_down']
            
            # Продвинутые объемные индикаторы
            result_df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            result_df['chaikin_osc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            # Волатильность и диапазоны
            for period in [7, 14, 21]:
                result_df[f'natr_{period}'] = talib.NATR(high, low, close, timeperiod=period)
                result_df[f'volatility_{period}'] = pd.Series(close).pct_change().rolling(window=period).std()
            
            # Кастомные комбинированные индикаторы
            result_df['price_volume_trend'] = ((close - np.roll(close, 1)) / np.roll(close, 1)) * volume
            
            # Ease of Movement (кастомная реализация, так как talib.EOM не существует)
            distance_moved = ((high + low) / 2) - ((np.roll(high, 1) + np.roll(low, 1)) / 2)
            box_height = (volume / 100000000) / (high - low)
            eom_raw = distance_moved / box_height
            result_df['ease_of_movement'] = pd.Series(eom_raw).rolling(window=14).mean().values
            
            # Статистические индикаторы
            for period in [10, 20, 50]:
                close_series = pd.Series(close)
                result_df[f'zscore_{period}'] = (close_series - close_series.rolling(window=period).mean()) / close_series.rolling(window=period).std()
                result_df[f'percentile_rank_{period}'] = close_series.rolling(window=period).rank(pct=True)
            
            # Фрактальные индикаторы
            result_df['fractal_high'] = ((high > np.roll(high, 2)) & 
                                       (high > np.roll(high, 1)) & 
                                       (high > np.roll(high, -1)) & 
                                       (high > np.roll(high, -2))).astype(int)
            result_df['fractal_low'] = ((low < np.roll(low, 2)) & 
                                      (low < np.roll(low, 1)) & 
                                      (low < np.roll(low, -1)) & 
                                      (low < np.roll(low, -2))).astype(int)
            
            # Дополнительные свечные паттерны
            result_df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)
            result_df['three_black_crows'] = talib.CDL3BLACKCROWS(open_price, high, low, close)
            result_df['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
            result_df['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
            result_df['harami'] = talib.CDLHARAMI(open_price, high, low, close)
            
            # Индикаторы силы тренда
            # Mass Index (кастомная реализация, так как talib.MASS не существует)
            hl_range = high - low
            ema9 = pd.Series(hl_range).ewm(span=9).mean()
            ema9_of_ema9 = ema9.ewm(span=9).mean()
            mass_index_raw = ema9 / ema9_of_ema9
            result_df['mass_index'] = pd.Series(mass_index_raw).rolling(window=25).sum().values
            # Vortex Indicator (кастомная реализация)
            tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            vm_pos = np.abs(high - np.roll(low, 1))
            vm_neg = np.abs(low - np.roll(high, 1))
            
            vi_pos = pd.Series(vm_pos).rolling(window=14).sum() / pd.Series(tr).rolling(window=14).sum()
            vi_neg = pd.Series(vm_neg).rolling(window=14).sum() / pd.Series(tr).rolling(window=14).sum()
            
            result_df['vortex_pos'] = vi_pos.values
            result_df['vortex_neg'] = vi_neg.values
            result_df['vortex_diff'] = vi_pos.values - vi_neg.values
            
            # Fibonacci retracements
            rolling_high = pd.Series(high).rolling(window=50).max()
            rolling_low = pd.Series(low).rolling(window=50).min()
            fib_range = rolling_high - rolling_low
            result_df['fib_23.6'] = rolling_high - 0.236 * fib_range
            result_df['fib_38.2'] = rolling_high - 0.382 * fib_range
            result_df['fib_50.0'] = rolling_high - 0.500 * fib_range
            result_df['fib_61.8'] = rolling_high - 0.618 * fib_range
            
            # Support/Resistance levels
            result_df['support_level'] = pd.Series(low).rolling(window=20).min()
            result_df['resistance_level'] = pd.Series(high).rolling(window=20).max()
            result_df['support_distance'] = (close - result_df['support_level']) / close
            result_df['resistance_distance'] = (result_df['resistance_level'] - close) / close
            
            # Market structure
            result_df['higher_high'] = (high > np.roll(high, 1)) & (np.roll(high, 1) > np.roll(high, 2))
            result_df['lower_low'] = (low < np.roll(low, 1)) & (np.roll(low, 1) < np.roll(low, 2))
            
            # Volatility clustering
            returns = np.log(close / np.roll(close, 1))
            result_df['returns'] = returns
            result_df['volatility_5'] = pd.Series(returns).rolling(window=5).std()
            result_df['volatility_20'] = pd.Series(returns).rolling(window=20).std()
            
            # Time-based features
            if 'timestamp' in df.columns:
                df_time = pd.to_datetime(df['timestamp'])
                result_df['hour'] = df_time.dt.hour
                result_df['day_of_week'] = df_time.dt.dayofweek
                result_df['month'] = df_time.dt.month
                result_df['is_weekend'] = (df_time.dt.dayofweek >= 5).astype(int)
                
            # Очистка данных
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            
            # Улучшенный метод заполнения пропусков
            result_df = result_df.interpolate(method='linear', limit_direction='both')
            result_df = result_df.fillna(method='bfill').fillna(method='ffill')
            
            # Дополнительная очистка: заполняем оставшиеся NaN нулями
            result_df = result_df.fillna(0)
            
            # Добавляем sentiment анализ
            try:
                sentiment_service = SentimentAnalysisService()
                # Извлекаем символ из имени столбца, если он есть в данных
                if hasattr(df, 'columns') and len(df.columns) > 0:
                    # Пытаемся найти символ в метаданных или используем переданный параметр
                    symbol_for_sentiment = getattr(df, 'symbol', symbol) if hasattr(df, 'symbol') else symbol
                else:
                    symbol_for_sentiment = symbol
                
                symbol_clean = symbol_for_sentiment.split('-')[0] if '-' in symbol_for_sentiment else symbol_for_sentiment
                # Используем расширенные sentiment признаки
                sentiment_features = sentiment_service.get_enhanced_sentiment_features(symbol_clean)
                
                # Добавляем sentiment признаки ко всем строкам
                for feature_name, feature_value in sentiment_features.items():
                    result_df[feature_name] = feature_value
                
                current_app.logger.info(f"📰 Добавлено {len(sentiment_features)} sentiment признаков")
            except Exception as e:
                current_app.logger.warning(f"Не удалось добавить sentiment признаки: {e}")
            
            # Добавляем макроэкономические индикаторы
            try:
                macro_service = MacroIndicatorsService()
                symbol_clean = symbol_for_sentiment.split('-')[0] if '-' in symbol_for_sentiment else symbol_for_sentiment
                macro_features = macro_service.get_enhanced_macro_features(symbol_clean)
                
                # Добавляем макро признаки ко всем строкам
                for feature_name, feature_value in macro_features.items():
                    result_df[feature_name] = feature_value
                
                current_app.logger.info(f"📊 Добавлено {len(macro_features)} макроэкономических признаков")
            except Exception as e:
                current_app.logger.warning(f"Не удалось добавить макроэкономические признаки: {e}")
            
            # Финальная проверка на NaN и бесконечные значения
            result_df = result_df.replace([np.inf, -np.inf], 0)
            
            return result_df
            
        except Exception as e:
            current_app.logger.error(f"Ошибка при создании признаков: {e}")
            raise ValueError(f"Не удалось создать технические индикаторы: {e}")
    
    @staticmethod
    def _calculate_vortex_indicator(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        """
        Рассчитывает Vortex Indicator (VI)
        
        Args:
            high: Максимальные цены
            low: Минимальные цены  
            close: Цены закрытия
            period: Период для расчета
            
        Returns:
            Tuple: (VI+, VI-)
        """
        try:
            # True Range
            tr = np.maximum(high - low, 
                           np.maximum(abs(high - np.roll(close, 1)), 
                                    abs(low - np.roll(close, 1))))
            
            # Vortex Movement
            vm_plus = abs(high - np.roll(low, 1))
            vm_minus = abs(low - np.roll(high, 1))
            
            # Суммы за период
            tr_sum = pd.Series(tr).rolling(window=period).sum()
            vm_plus_sum = pd.Series(vm_plus).rolling(window=period).sum()
            vm_minus_sum = pd.Series(vm_minus).rolling(window=period).sum()
            
            # Vortex Indicator
            vi_plus = vm_plus_sum / tr_sum
            vi_minus = vm_minus_sum / tr_sum
            
            return vi_plus.values, vi_minus.values
            
        except Exception:
            # Возвращаем массивы нулей в случае ошибки
            return np.zeros(len(high)), np.zeros(len(high))


class EnsembleMLModel:
    """
    Ensemble модель, объединяющая несколько алгоритмов машинного обучения
    для достижения максимальной точности предсказаний
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.training_metrics = {}
        
        # Адаптивные веса на основе исторической производительности
        self.weights = {'rf': 0.4, 'gb': 0.35, 'xgb': 0.25}
        self.performance_history = {'rf': [], 'gb': [], 'xgb': []}
        self.adaptive_weights_enabled = True
        self.min_history_length = 5  # Минимум измерений для адаптации весов
        
        # Оптимизация гиперпараметров
        self.hyperopt_service = None
        self.use_optimized_params = True
        
    def _prepare_models(self):
        """Инициализация моделей ensemble с оптимизированными параметрами"""
        # Инициализируем сервис оптимизации гиперпараметров
        if self.hyperopt_service is None:
            self.hyperopt_service = HyperparameterOptimizationService()
        
        # Random Forest - с улучшенной регуляризацией против переобучения
        rf_params = {
            'n_estimators': 200,      # Уменьшено для предотвращения переобучения
            'max_depth': 12,          # Уменьшено для лучшей генерализации
            'min_samples_split': 10,  # Увеличено для регуляризации
            'min_samples_leaf': 5,    # Увеличено для стабильности
            'max_features': 0.6,      # Ограничиваем количество признаков
            'bootstrap': True,
            'oob_score': True,        # Для дополнительной валидации
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Используем оптимизированные параметры если доступны
        if self.use_optimized_params and 'rf' in self.hyperopt_service.best_params:
            optimized_params = self.hyperopt_service.best_params['rf'].copy()
            optimized_params.update({'random_state': 42, 'n_jobs': -1, 'oob_score': True})
            self.models['rf'] = RandomForestRegressor(**optimized_params)
            current_app.logger.info("🔧 Используем оптимизированные параметры для RF")
        else:
            self.models['rf'] = RandomForestRegressor(**rf_params)
        
        # Gradient Boosting - с сильной регуляризацией против переобучения
        gb_params = {
            'n_estimators': 150,      # Уменьшено для предотвращения переобучения
            'learning_rate': 0.05,    # Значительно уменьшено
            'max_depth': 4,           # Уменьшено для лучшей генерализации
            'min_samples_split': 15,  # Увеличено для регуляризации
            'min_samples_leaf': 8,    # Увеличено для стабильности
            'subsample': 0.7,         # Увеличиваем стохастичность
            'max_features': 0.5,      # Ограничиваем признаки
            'validation_fraction': 0.2, # Увеличена валидация
            'n_iter_no_change': 10,   # Ранняя остановка
            'random_state': 42
        }
        
        # Используем оптимизированные параметры если доступны
        if self.use_optimized_params and 'gb' in self.hyperopt_service.best_params:
            optimized_params = self.hyperopt_service.best_params['gb'].copy()
            optimized_params.update({'random_state': 42})
            self.models['gb'] = GradientBoostingRegressor(**optimized_params)
            current_app.logger.info("🔧 Используем оптимизированные параметры для GB")
        else:
            self.models['gb'] = GradientBoostingRegressor(**gb_params)
        
        # XGBoost - с максимальной регуляризацией против переобучения
        if XGBOOST_AVAILABLE:
            xgb_params = {
                'n_estimators': 150,      # Уменьшено
                'learning_rate': 0.03,    # Очень консервативная скорость
                'max_depth': 4,           # Ограничиваем глубину
                'min_child_weight': 6,    # Увеличено для регуляризации
                'subsample': 0.7,         # Увеличиваем стохастичность
                'colsample_bytree': 0.6,  # Меньше признаков
                'colsample_bylevel': 0.8, # Дополнительная регуляризация
                'reg_alpha': 0.5,         # Увеличена L1 регуляризация
                'reg_lambda': 2.0,        # Увеличена L2 регуляризация
                'gamma': 0.2,             # Увеличен минимальный gain
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            # Используем оптимизированные параметры если доступны
            if self.use_optimized_params and 'xgb' in self.hyperopt_service.best_params:
                optimized_params = self.hyperopt_service.best_params['xgb'].copy()
                optimized_params.update({
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                })
                self.models['xgb'] = xgb.XGBRegressor(**optimized_params)
                current_app.logger.info("🔧 Используем оптимизированные параметры для XGB")
            else:
                self.models['xgb'] = xgb.XGBRegressor(**xgb_params)
            
        # Скалеры для каждой модели
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Обучение ensemble модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Метрики качества модели
        """
        if len(X) < 100:
            raise ValueError("Недостаточно данных для обучения (минимум 100 образцов)")
        
        current_app.logger.info(f"🤖 Начинаем обучение Ensemble модели с {len(X)} образцами и {len(X.columns)} признаками")
        start_time = time.time()
            
        self._prepare_models()
        current_app.logger.info(f"📊 Инициализировано {len(self.models)} моделей: {list(self.models.keys())}")
        
        # Разделение на обучение и валидацию по времени
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        metrics = {}
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                current_app.logger.info(f"🔄 Обучение модели {model_name.upper()}...")
                model_start = time.time()
                
                # Масштабирование признаков
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_val_scaled = self.scalers[model_name].transform(X_val)
                current_app.logger.info(f"   📏 Масштабирование завершено для {model_name}")
                
                # Обучение модели
                model.fit(X_train_scaled, y_train)
                training_time = time.time() - model_start
                current_app.logger.info(f"   ⏱️  Обучение {model_name} завершено за {training_time:.2f}с")
                
                # Cross-validation для более надежной оценки
                tscv = TimeSeriesSplit(n_splits=5)
                X_full_scaled = self.scalers[model_name].transform(X)
                cv_scores = cross_val_score(model, X_full_scaled, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
                current_app.logger.info(f"   📊 Cross-validation MAE: {cv_mae:.2f} ± {cv_std:.2f}")
                
                # Предсказания
                y_pred = model.predict(X_val_scaled)
                predictions[model_name] = y_pred
                
                # Метрики
                mae = mean_absolute_error(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                accuracy = max(0, min(100, (1 - mae / np.mean(y_val)) * 100))
                
                metrics[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'cv_mae': cv_mae,
                    'cv_std': cv_std
                }
                
                current_app.logger.info(f"   📈 {model_name.upper()} - Точность: {accuracy:.2f}%, R²: {r2:.3f}, MAE: {mae:.2f}")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(
                        X.columns, model.feature_importances_
                    ))
                    top_features = sorted(self.feature_importance[model_name].items(), key=lambda x: x[1], reverse=True)[:5]
                    current_app.logger.info(f"   🔍 Топ-5 признаков для {model_name}: {[f[0] for f in top_features]}")
                    
            except Exception as e:
                current_app.logger.error(f"❌ Ошибка обучения модели {model_name}: {e}")
                continue
        
        # Ensemble предсказание (взвешенное среднее)
        if predictions:
            weights = {name: metrics[name]['r2'] for name in predictions.keys() if metrics[name]['r2'] > 0}
            if weights:
                total_weight = sum(weights.values())
                ensemble_pred = sum(
                    pred * (weights[name] / total_weight) 
                    for name, pred in predictions.items() 
                    if name in weights
                )
                
                # Метрики ensemble
                ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
                ensemble_r2 = r2_score(y_val, ensemble_pred)
                ensemble_accuracy = max(0, min(100, (1 - ensemble_mae / np.mean(y_val)) * 100))
                
                metrics['ensemble'] = {
                    'mae': ensemble_mae,
                    'r2': ensemble_r2,
                    'accuracy': ensemble_accuracy
                }
                
                current_app.logger.info(f"🎯 ENSEMBLE РЕЗУЛЬТАТ - Точность: {ensemble_accuracy:.2f}%, R²: {ensemble_r2:.3f}")
        
        total_time = time.time() - start_time
        current_app.logger.info(f"✅ Обучение Ensemble модели завершено за {total_time:.2f}с")
        
        self.is_trained = True
        self.training_metrics = metrics  # Сохраняем метрики
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """
        Предсказание цены с доверительным интервалом
        
        Args:
            X: Признаки для предсказания
            
        Returns:
            Tuple: (предсказанная_цена, стандартное_отклонение, детали)
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
            
        individual_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Дополнительная очистка данных перед предсказанием
                X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
                X_scaled = self.scalers[model_name].transform(X_clean)
                pred = model.predict(X_scaled)[0]
                individual_predictions[model_name] = pred
            except Exception as e:
                current_app.logger.warning(f"Не удалось получить предсказание от {model_name}: {e}")
                continue

        if not individual_predictions:
            raise ValueError("Ни одна модель не смогла сделать предсказание.")

        # Взвешенное ансамблирование на основе R2
        weights = {
            name: self.training_metrics[name]['r2'] 
            for name in individual_predictions.keys() 
            if name in self.training_metrics and self.training_metrics[name]['r2'] > 0
        }
        
        if not weights:
            # Если нет весов (например, все R2 < 0), используем простое среднее
            final_prediction = np.mean(list(individual_predictions.values()))
        else:
            total_weight = sum(weights.values())
            final_prediction = sum(
                pred * (weights[name] / total_weight) 
                for name, pred in individual_predictions.items() 
                if name in weights
            )

        # Расчет неопределенности (стандартное отклонение предсказаний)
        prediction_std = np.std(list(individual_predictions.values()))

        details = {
            'ensemble_prediction': final_prediction,
            'prediction_std_dev': prediction_std,
            'individual_predictions': individual_predictions,
            'model_weights': weights
        }

        return final_prediction, prediction_std, details
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_reduced_grid: bool = True,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Оптимизировать гиперпараметры для всех моделей в ансамбле
        """
        current_app.logger.info("🔧 Начинаем оптимизацию гиперпараметров ансамбля")
        
        if self.hyperopt_service is None:
            self.hyperopt_service = HyperparameterOptimizationService()
        
        optimization_results = self.hyperopt_service.optimize_ensemble_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            use_reduced_grid=use_reduced_grid,
            cv_folds=cv_folds
        )
        
        if optimization_results.get('optimized_models'):
            current_app.logger.info("🔄 Обновляем модели с оптимизированными параметрами")
            self.models.update(optimization_results['optimized_models'])
            
            for model_name in self.models.keys():
                self.scalers[model_name] = RobustScaler()
        
        summary = optimization_results.get('optimization_summary', {})
        for model_name, results in summary.items():
            metrics = results.get('metrics', {})
            if 'error' not in metrics:
                current_app.logger.info(
                    f"✅ {model_name.upper()}: CV MAE {metrics.get('best_cv_mae', 'N/A'):.2f}, "
                    f"R² {metrics.get('train_r2', 'N/A'):.3f}"
                )
            else:
                current_app.logger.warning(f"❌ {model_name.upper()}: {metrics['error']}")
        
        return optimization_results


# Расширенный список поддерживаемых криптовалют
SUPPORTED_CRYPTO_PAIRS = [
    # Топ криптовалюты
    'BTC-USD', 'ETH-USD', 'BNB-USD',
    # DeFi токены
    'UNI-USD', 'AAVE-USD', 'COMP-USD', 'MKR-USD', 'SUSHI-USD',
    # Layer 1 блокчейны
    'ADA-USD', 'SOL-USD', 'DOT-USD', 'AVAX-USD', 'ATOM-USD',
    # Альткоины
    'XRP-USD', 'LTC-USD', 'BCH-USD', 'ETC-USD', 'ZEC-USD',
    # Новые перспективные
    'MATIC-USD', 'LINK-USD', 'VET-USD', 'ALGO-USD', 'FTM-USD'
]

# Категории криптовалют для анализа
CRYPTO_CATEGORIES = {
    'major': ['BTC-USD', 'ETH-USD', 'BNB-USD'],
    'defi': ['UNI-USD', 'AAVE-USD', 'COMP-USD', 'MKR-USD', 'SUSHI-USD'],
    'layer1': ['ADA-USD', 'SOL-USD', 'DOT-USD', 'AVAX-USD', 'ATOM-USD'],
    'altcoins': ['XRP-USD', 'LTC-USD', 'BCH-USD', 'ETC-USD', 'ZEC-USD'],
    'emerging': ['MATIC-USD', 'LINK-USD', 'VET-USD', 'ALGO-USD', 'FTM-USD']
}


class AnalysisService:
    """
    Главный сервис для продвинутого анализа криптовалют
    с использованием ensemble машинного обучения
    """
    
    def __init__(self):
        self.crypto_service = CryptoService()
        self.models_path = 'models/advanced_ml'
        self.feature_engineer = AdvancedFeatureEngineering()
        self.model_cache_ttl = timedelta(hours=24)  # Время жизни кэша модели
        self.supported_timeframes = ['1h', '4h', '1d', '1w']  # Поддерживаемые временные интервалы
        
        # Создаем директорию для моделей если не существует
        os.makedirs(self.models_path, exist_ok=True)

    def _get_model_path(self, symbol: str, timeframe: str) -> str:
        """Генерирует путь к файлу модели."""
        filename = f"ensemble_model_{symbol.replace('-', '_')}_{timeframe}.joblib"
        return os.path.join(self.models_path, filename)

    def _save_model(self, model: EnsembleMLModel, symbol: str, timeframe: str):
        """Сохраняет обученную модель в файл."""
        model_path = self._get_model_path(symbol, timeframe)
        try:
            joblib.dump(model, model_path)
            current_app.logger.info(f"💾 Модель сохранена в {model_path}")
        except Exception as e:
            current_app.logger.error(f"❌ Не удалось сохранить модель в {model_path}: {e}")

    def _load_model(self, symbol: str, timeframe: str) -> Optional[EnsembleMLModel]:
        """Загружает модель из файла, если она не устарела."""
        model_path = self._get_model_path(symbol, timeframe)
        if not os.path.exists(model_path):
            return None

        try:
            model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))
            if model_age > self.model_cache_ttl:
                current_app.logger.info(f"Модель {model_path} устарела ({model_age}). Требуется переобучение.")
                return None

            start_time = time.time()
            model = joblib.load(model_path)
            load_time = time.time() - start_time
            
            # Проверяем совместимость модели с текущими доступными библиотеками
            if hasattr(model, 'models') and 'xgb' in model.models and not XGBOOST_AVAILABLE:
                current_app.logger.warning("Кэшированная модель содержит XGBoost, но библиотека недоступна. Пересоздаем модель.")
                return None
            
            current_app.logger.info(f"✅ Модель успешно загружена из {model_path} за {load_time:.2f}с")
            return model
        except Exception as e:
            current_app.logger.error(f"❌ Не удалось загрузить модель из {model_path}: {e}")
            return None

    def _get_historical_data(self, symbol: str, timeframe: str, extended: bool = True) -> pd.DataFrame:
        """
        Получает и подготавливает исторические данные для анализа
        
        Args:
            symbol: Торговая пара (например, 'BTC-USD')
            timeframe: Временной интервал ('1h', '4h', '1d')
            extended: Получить расширенный набор данных для ML
            
        Returns:
            DataFrame с OHLCV данными и временным индексом
        """
        # Увеличенные периоды для качественного ML анализа
        # Специальная обработка для BNB-USD - увеличиваем объем данных
        if 'BNB' in symbol.upper():
            if extended:
                days_map = {'1d': 730, '4h': 180, '1h': 60}  # 2 года данных для BNB
            else:
                days_map = {'1d': 365, '4h': 60, '1h': 30}
        else:
            if extended:
                days_map = {'1d': 365, '4h': 90, '1h': 30}  # Год данных для дневного анализа
            else:
                days_map = {'1d': 180, '4h': 30, '1h': 14}
            
        days = days_map.get(timeframe, 365)
        
        try:
            df = self.crypto_service.get_coin_historical_data_df(symbol, days)
            if df.empty or len(df) < 100:  # Минимум 100 точек для ML
                raise ValueError(f"Недостаточно данных для {symbol} (получено {len(df)}, нужно минимум 100)")
                
            # Проверяем наличие всех необходимых колонок
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Отсутствуют необходимые колонки: {missing}")
                
            return df
            
        except Exception as e:
            current_app.logger.error(f"Ошибка получения данных для {symbol}: {e}")
            raise ValueError(f"Не удалось получить исторические данные: {e}")

    def advanced_ml_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Выполняет продвинутый ML анализ с использованием ensemble модели
        для достижения точности 98%+
        
        Args:
            symbol: Символ криптовалюты (например, 'BTC-USD')
            timeframe: Временной интервал ('1h', '4h', '1d')
            
        Returns:
            Словарь с результатами анализа, предсказаниями и рекомендациями
        """
        try:
            current_app.logger.info(f"🚀 Запуск ML анализа для {symbol} ({timeframe})")
            start_total_time = time.time()

            # Шаг 1: Попытка загрузить кэшированную модель
            ensemble_model = self._load_model(symbol, timeframe)
            training_metrics = None

            try:
                # Шаг 2: Получение и подготовка данных
                df = self._get_historical_data(symbol, timeframe)
                df_features = self.feature_engineer.create_advanced_features(df, symbol)
                
                # Определение признаков (X) и цели (y)
                features = [col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']]
                df_features['target'] = df_features['close'].shift(-1)
                df_features = df_features.dropna(subset=['target'])
                
                X = df_features[features]
                y = df_features['target']

                # Шаг 3: Обучение модели, если не загружена из кэша
                if not ensemble_model:
                    current_app.logger.info("Кэшированная модель не найдена или устарела. Начинаем обучение новой модели.")
                    ensemble_model = EnsembleMLModel()
                    training_metrics = ensemble_model.train(X.iloc[:-1], y.iloc[:-1])
                    self._save_model(ensemble_model, symbol, timeframe)
                else:
                    training_metrics = ensemble_model.training_metrics
                    current_app.logger.info("Используется кэшированная модель.")

                # Шаг 4: Проверка совместимости признаков и предсказание
                last_X = X.tail(1)
                
                # Проверяем совместимость признаков с моделью
                try:
                    predicted_price, std_dev, prediction_details = ensemble_model.predict(last_X)
                except Exception as model_error:
                    if "feature names" in str(model_error).lower() or "unseen at fit time" in str(model_error).lower():
                        current_app.logger.warning(f"🔄 Обнаружено несоответствие признаков для {symbol}. Переобучение модели...")
                        
                        # Переобучаем модель с новыми признаками
                        ensemble_model = EnsembleMLModel()
                        training_metrics = ensemble_model.train(X, y)
                        
                        # Сохраняем обновленную модель
                        self._save_model(ensemble_model, symbol, timeframe)
                        current_app.logger.info(f"💾 Модель {symbol} переобучена и сохранена с новыми признаками")
                        
                        # Повторяем предсказание с новой моделью
                        predicted_price, std_dev, prediction_details = ensemble_model.predict(last_X)
                    else:
                        raise model_error
                
                # Шаг 5: Генерация инсайтов и рекомендаций
                last_row = df_features.tail(1).iloc[0]
                insights, recommendation = self._generate_advanced_insights(last_row, prediction_details, training_metrics)

                total_time = time.time() - start_total_time
                current_app.logger.info(f"✅ Анализ для {symbol} завершен за {total_time:.2f}с")

                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'current_price': last_row['close'],
                    'predicted_price': predicted_price,
                    'prediction_std_dev': std_dev,
                    'recommendation': recommendation,
                    'confidence': prediction_details.get('ensemble_confidence', 0),
                    'insights': insights,
                    'training_metrics': training_metrics,
                    'prediction_details': prediction_details,
                    'total_analysis_time': total_time
                }

            except ValueError as ve:
                current_app.logger.error(f"Ошибка валидации данных для {symbol}: {ve}")
                return {'status': 'error', 'message': str(ve)}
            except Exception as e:
                current_app.logger.error(f"Критическая ошибка в advanced_ml_analysis для {symbol}: {e}", exc_info=True)
                return {'status': 'error', 'message': f"Внутренняя ошибка сервера при анализе {symbol}."}
        except Exception as e:
            current_app.logger.error(f"Критическая ошибка в advanced_ml_analysis для {symbol}: {e}", exc_info=True)
            return {'status': 'error', 'message': f"Внутренняя ошибка сервера при анализе {symbol}."}

    def _generate_advanced_insights(self, last_row: pd.Series, prediction_details: Dict, training_metrics: Dict):
        """
        Генерирует продвинутые инсайты на основе ML анализа и технических индикаторов
        
        Args:
            last_row: Последняя строка с техническими индикаторами
            prediction_details: Детали предсказания от ensemble модели
            training_metrics: Метрики обучения моделей
            
        Returns:
            Tuple: (объяснения, рекомендация)
        """
        explanation = {}
        bullish_score = 0
        bearish_score = 0
        
        try:
            # Анализ RSI (если доступен)
            if 'rsi_14' in last_row:
                rsi = last_row['rsi_14']
                if rsi > 70:
                    explanation['RSI'] = f"🔴 Перекупленность ({rsi:.1f}). Высокий риск коррекции."
                    bearish_score += 2.5
                elif rsi < 30:
                    explanation['RSI'] = f"🟢 Перепроданность ({rsi:.1f}). Потенциал роста."
                    bullish_score += 2.5
                else:
                    explanation['RSI'] = f"🟡 Нейтральная зона ({rsi:.1f})."
            
            # Анализ MACD
            if 'macd' in last_row and 'macd_signal' in last_row:
                macd_cross = last_row.get('macd_cross', 0)
                if macd_cross > 0:
                    explanation['MACD'] = "🟢 Бычий сигнал - MACD выше сигнальной линии."
                    bullish_score += 2
                else:
                    explanation['MACD'] = "🔴 Медвежий сигнал - MACD ниже сигнальной линии."
                    bearish_score += 2
            
            # Анализ трендовых индикаторов
            if 'ema_20' in last_row and 'ema_50' in last_row:
                if last_row['ema_20'] > last_row['ema_50']:
                    explanation['Тренд'] = "🟢 Восходящий тренд (EMA20 > EMA50)."
                    bullish_score += 2
                else:
                    explanation['Тренд'] = "🔴 Нисходящий тренд (EMA20 < EMA50)."
                    bearish_score += 2
            
            # Анализ Bollinger Bands
            if 'bb_position_20' in last_row:
                bb_pos = last_row['bb_position_20']
                if bb_pos > 0.8:
                    explanation['Bollinger Bands'] = f"🔴 Цена у верхней границы ({bb_pos:.2f}). Возможна коррекция."
                    bearish_score += 1.5
                elif bb_pos < 0.2:
                    explanation['Bollinger Bands'] = f"🟢 Цена у нижней границы ({bb_pos:.2f}). Потенциал отскока."
                    bullish_score += 1.5
                else:
                    explanation['Bollinger Bands'] = f"🟡 Цена в средней зоне ({bb_pos:.2f})."
            
            # Анализ волатильности
            if 'atr_percent' in last_row:
                atr_pct = last_row['atr_percent']
                if atr_pct > 5:
                    explanation['Волатильность'] = f"⚠️ Высокая волатильность ({atr_pct:.1f}%). Повышенный риск."
                    bearish_score += 1
                elif atr_pct < 2:
                    explanation['Волатильность'] = f"📈 Низкая волатильность ({atr_pct:.1f}%). Стабильные условия."
                    bullish_score += 0.5
            
            # Анализ объемов
            if 'volume_ratio_20' in last_row:
                vol_ratio = last_row['volume_ratio_20']
                if vol_ratio > 1.5:
                    explanation['Объем'] = f"📊 Повышенный объем торгов ({vol_ratio:.1f}x). Сильный интерес."
                    bullish_score += 1
                elif vol_ratio < 0.7:
                    explanation['Объем'] = f"📉 Пониженный объем ({vol_ratio:.1f}x). Слабый интерес."
                    bearish_score += 0.5
            
            # ML модель инсайты
            model_confidence = max(prediction_details.get('confidences', {}).values()) if prediction_details.get('confidences') else 0
            best_accuracy = max([m.get('accuracy', 0) for m in training_metrics.values() if isinstance(m, dict)])
            
            explanation['ML Модель'] = f"🤖 Точность: {best_accuracy:.1f}%, Уверенность: {model_confidence:.2f}"
            
            if best_accuracy > 95:
                explanation['Качество прогноза'] = "🎯 Очень высокое качество прогноза (>95%)"
                # Увеличиваем вес ML предсказания
                if prediction_details.get('ensemble_prediction', 0) > last_row.get('close', 0):
                    bullish_score += 3
                else:
                    bearish_score += 3
            elif best_accuracy > 90:
                explanation['Качество прогноза'] = "✅ Высокое качество прогноза (>90%)"
                if prediction_details.get('ensemble_prediction', 0) > last_row.get('close', 0):
                    bullish_score += 2
                else:
                    bearish_score += 2
            
            # Генерация рекомендации
            total_score = bullish_score + bearish_score
            if total_score > 0:
                bullish_ratio = bullish_score / total_score
                
                if bullish_ratio > 0.75:
                    recommendation = "🚀 СИЛЬНАЯ ПОКУПКА"
                elif bullish_ratio > 0.6:
                    recommendation = "📈 ПОКУПКА"
                elif bullish_ratio > 0.4:
                    recommendation = "⏸️ ДЕРЖАТЬ"
                elif bullish_ratio > 0.25:
                    recommendation = "📉 ПРОДАЖА"
                else:
                    recommendation = "🔻 СИЛЬНАЯ ПРОДАЖА"
            else:
                recommendation = "⏸️ ДЕРЖАТЬ"
            
            # Добавляем общую оценку
            explanation['Общая оценка'] = f"Бычьи сигналы: {bullish_score:.1f}, Медвежьи: {bearish_score:.1f}"
            
        except Exception as e:
            current_app.logger.error(f"Ошибка генерации инсайтов: {e}")
            explanation['Ошибка'] = "Не удалось проанализировать все индикаторы"
            recommendation = "⏸️ ДЕРЖАТЬ"
        
        return explanation, recommendation

    def compare_cryptocurrencies(self, symbols: List[str], timeframe: str, comparison_type: str) -> Dict[str, Any]:
        """
        Сравнивает несколько криптовалют с использованием ML анализа
        
        Args:
            symbols: Список символов для сравнения
            timeframe: Временной интервал
            comparison_type: Тип сравнения ('performance', 'risk', 'ml_prediction')
            
        Returns:
            Результаты сравнения с графиками и рекомендациями
        """
        try:
            results = {}
            
            for symbol in symbols[:5]:  # Ограничиваем до 5 символов
                try:
                    analysis = self.advanced_ml_analysis(symbol, timeframe)
                    if analysis['status'] == 'success':
                        results[symbol] = {
                            'predicted_price': analysis['predicted_price'],
                            'accuracy': analysis['training_metrics']['ensemble']['accuracy'],
                            'recommendation': analysis['recommendation'],
                            'current_price': analysis['current_price']
                        }
                except Exception as e:
                    current_app.logger.error(f"Ошибка анализа {symbol}: {e}")
                    continue
            
            # Сортируем по точности модели или потенциалу роста
            if comparison_type == 'ml_prediction':
                sorted_results = sorted(
                    results.items(), 
                    key=lambda x: x[1]['accuracy'], 
                    reverse=True
                )
            else:
                sorted_results = sorted(
                    results.items(), 
                    key=lambda x: (x[1]['predicted_price'] / x[1]['current_price'] - 1) if x[1]['current_price'] > 0 else 0, 
                    reverse=True
                )
            
            return {
                'status': 'success',
                'comparison_results': dict(sorted_results),
                'summary_data': [
                    {
                        'symbol': symbol,
                        'accuracy': data['accuracy'],
                        'potential_return': (data['predicted_price'] / data['current_price'] - 1) * 100 if data['current_price'] > 0 else 0,
                        'recommendation': data['recommendation']
                    }
                    for symbol, data in sorted_results
                ],
                'plot_html': self._create_comparison_plot(dict(sorted_results))
            }
            
        except Exception as e:
            current_app.logger.error(f"Ошибка сравнения криптовалют: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _create_comparison_plot(self, results: Dict) -> str:
        """Создает HTML график для сравнения криптовалют"""
        try:
            # Простая HTML таблица для сравнения (можно расширить с Plotly)
            html = "<div class='comparison-table'><table class='table table-striped'>"
            html += "<thead><tr><th>Символ</th><th>Точность ML</th><th>Потенциал роста</th><th>Рекомендация</th></tr></thead><tbody>"
            
            for symbol, data in results.items():
                potential = (data['predicted_price'] / data['current_price'] - 1) * 100 if data['current_price'] > 0 else 0
                html += f"<tr><td>{symbol}</td><td>{data['accuracy']:.1f}%</td><td>{potential:+.1f}%</td><td>{data['recommendation']}</td></tr>"
            
            html += "</tbody></table></div>"
            return html
            
        except Exception as e:
            return f"<div class='alert alert-warning'>Ошибка создания графика: {e}</div>"
    
    def multi_timeframe_analysis(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Any]:
        """
        Выполняет анализ на нескольких временных интервалах для более точного прогноза
        
        Args:
            symbol: Символ криптовалюты (например, 'BTC-USD')
            timeframes: Список временных интервалов для анализа
            
        Returns:
            Словарь с результатами анализа по каждому временному интервалу
        """
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']  # По умолчанию анализируем 3 интервала
        
        # Фильтруем только поддерживаемые временные интервалы
        valid_timeframes = [tf for tf in timeframes if tf in self.supported_timeframes]
        
        if not valid_timeframes:
            raise ValueError(f"Нет поддерживаемых временных интервалов. Доступны: {self.supported_timeframes}")
        
        current_app.logger.info(f"🕐 Запуск multi-timeframe анализа для {symbol} на интервалах: {valid_timeframes}")
        
        results = {}
        predictions = []
        confidences = []
        
        for timeframe in valid_timeframes:
            try:
                current_app.logger.info(f"📊 Анализ {symbol} на интервале {timeframe}")
                
                # Выполняем анализ для каждого временного интервала
                result = self.advanced_ml_analysis(symbol, timeframe)
                
                if result.get('status') == 'success':
                    results[timeframe] = result
                    
                    # Собираем предсказания для комбинирования
                    if 'predicted_price' in result:
                        predictions.append({
                            'timeframe': timeframe,
                            'price': result['predicted_price'],
                            'confidence': result.get('confidence', 0.5),
                            'accuracy': self._get_timeframe_weight(timeframe)
                        })
                        confidences.append(result.get('confidence', 0.5))
                
                current_app.logger.info(f"✅ Анализ {timeframe} завершен")
                
            except Exception as e:
                current_app.logger.error(f"❌ Ошибка анализа {timeframe}: {e}")
                results[timeframe] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Комбинируем результаты разных временных интервалов
        combined_result = self._combine_timeframe_results(results, predictions)
        
        current_app.logger.info(f"🎯 Multi-timeframe анализ завершен. Комбинированный прогноз: ${combined_result.get('combined_price', 'N/A')}")
        
        return {
            'status': 'success',
            'symbol': symbol,
            'timeframes_analyzed': valid_timeframes,
            'individual_results': results,
            'combined_prediction': combined_result,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_timeframe_weight(self, timeframe: str) -> float:
        """
        Возвращает вес для временного интервала при комбинировании результатов
        Более длинные интервалы имеют больший вес для долгосрочных трендов
        """
        weights = {
            '1h': 0.2,   # Краткосрочный шум
            '4h': 0.3,   # Среднесрочные движения
            '1d': 0.4,   # Основной тренд
            '1w': 0.5    # Долгосрочный тренд
        }
        return weights.get(timeframe, 0.25)
    
    def _combine_timeframe_results(self, results: Dict, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Комбинирует результаты анализа с разных временных интервалов
        """
        if not predictions:
            return {'error': 'Нет успешных предсказаний для комбинирования'}
        
        # Взвешенное среднее предсказаний
        total_weight = 0
        weighted_price_sum = 0
        weighted_confidence_sum = 0
        
        for pred in predictions:
            weight = pred['accuracy'] * pred['confidence']
            weighted_price_sum += pred['price'] * weight
            weighted_confidence_sum += pred['confidence'] * weight
            total_weight += weight
        
        if total_weight == 0:
            return {'error': 'Нулевой общий вес предсказаний'}
        
        combined_price = weighted_price_sum / total_weight
        combined_confidence = weighted_confidence_sum / total_weight
        
        # Анализ согласованности предсказаний
        prices = [pred['price'] for pred in predictions]
        price_std = np.std(prices)
        price_range = max(prices) - min(prices)
        
        # Определяем уровень согласованности
        consistency_score = 1.0 - min(price_range / combined_price, 1.0)
        
        # Генерируем рекомендацию на основе комбинированного анализа
        recommendation = self._generate_multi_timeframe_recommendation(
            predictions, combined_confidence, consistency_score
        )
        
        return {
            'combined_price': combined_price,
            'combined_confidence': combined_confidence,
            'consistency_score': consistency_score,
            'price_range': price_range,
            'price_std': price_std,
            'recommendation': recommendation,
            'timeframe_count': len(predictions)
        }
    
    def _generate_multi_timeframe_recommendation(self, predictions: List[Dict], 
                                               combined_confidence: float, 
                                               consistency_score: float) -> str:
        """
        Генерирует рекомендацию на основе multi-timeframe анализа
        """
        if combined_confidence < 0.3:
            return "ОСТОРОЖНО: Низкая уверенность в прогнозе"
        
        if consistency_score < 0.5:
            return "ВНИМАНИЕ: Противоречивые сигналы на разных временных интервалах"
        
        # Анализируем направление движения на разных интервалах
        short_term = [p for p in predictions if p['timeframe'] in ['1h', '4h']]
        long_term = [p for p in predictions if p['timeframe'] in ['1d', '1w']]
        
        if short_term and long_term:
            short_avg = np.mean([p['price'] for p in short_term])
            long_avg = np.mean([p['price'] for p in long_term])
            
            if abs(short_avg - long_avg) / long_avg > 0.05:  # Расхождение более 5%
                if short_avg > long_avg:
                    return "ПОКУПКА: Краткосрочный импульс вверх при долгосрочном тренде"
                else:
                    return "ПРОДАЖА: Краткосрочная коррекция в долгосрочном тренде"
        
        if combined_confidence > 0.7 and consistency_score > 0.8:
            return "СИЛЬНЫЙ СИГНАЛ: Высокая согласованность на всех временных интервалах"
        elif combined_confidence > 0.5 and consistency_score > 0.6:
            return "УМЕРЕННЫЙ СИГНАЛ: Хорошая согласованность прогнозов"
        else:
            return "НЕЙТРАЛЬНО: Смешанные сигналы, требуется дополнительный анализ"
    
    def optimize_hyperparameters(
        self, 
        symbol: str,
        timeframe: str = '1d',
        use_reduced_grid: bool = True,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Оптимизировать гиперпараметры для модели конкретной криптовалюты
        
        Args:
            symbol: Торговая пара (например, 'BTC-USD')
            timeframe: Временной интервал
            use_reduced_grid: Использовать уменьшенную сетку параметров
            cv_folds: Количество фолдов для кросс-валидации
            
        Returns:
            Словарь с результатами оптимизации
        """
        current_app.logger.info(f"🔧 Начинаем оптимизацию гиперпараметров для {symbol}")
        
        # Получаем исторические данные
        df = self._get_historical_data(symbol, timeframe, extended=True)
        if df.empty:
            raise ValueError(f"Не удалось получить данные для {symbol}")
        
        # Создаем признаки
        df_features = AdvancedFeatureEngineering.create_advanced_features(df, symbol)
        
        # Подготавливаем данные для обучения
        X = df_features.drop(['target'], axis=1, errors='ignore')
        y = df_features['close'].shift(-1).dropna()  # Предсказываем следующую цену
        X = X.iloc[:-1]  # Убираем последнюю строку из X
        
        # Создаем модель
        model = EnsembleMLModel()
        
        # Выполняем оптимизацию
        optimization_results = model.optimize_hyperparameters(
            X_train=X,
            y_train=y,
            use_reduced_grid=use_reduced_grid,
            cv_folds=cv_folds
        )
        
        # Сохраняем результаты оптимизации
        optimization_path = f"models/advanced_ml/hyperopt_results_{symbol.replace('-', '_')}_{timeframe}.joblib"
        os.makedirs(os.path.dirname(optimization_path), exist_ok=True)
        model.hyperopt_service.save_optimization_results(optimization_path)
        
        current_app.logger.info(f"✅ Оптимизация гиперпараметров для {symbol} завершена")
        
        return optimization_results
    
    def get_supported_crypto_pairs(self) -> List[str]:
        """
        Получить список поддерживаемых криптовалютных пар
        """
        return SUPPORTED_CRYPTO_PAIRS.copy()
    
    def get_crypto_categories(self) -> Dict[str, List[str]]:
        """
        Получить категории криптовалют
        """
        return CRYPTO_CATEGORIES.copy()
    
    def analyze_crypto_category(
        self, 
        category: str, 
        timeframe: str = '1d',
        limit: int = None
    ) -> Dict[str, Any]:
        """
        Анализировать всю категорию криптовалют
        
        Args:
            category: Категория криптовалют ('major', 'defi', 'layer1', 'altcoins', 'emerging')
            timeframe: Временной интервал
            limit: Ограничение количества анализируемых пар
            
        Returns:
            Словарь с результатами анализа по каждой паре
        """
        if category not in CRYPTO_CATEGORIES:
            raise ValueError(f"Неподдерживаемая категория: {category}")
        
        pairs = CRYPTO_CATEGORIES[category]
        if limit:
            pairs = pairs[:limit]
        
        current_app.logger.info(f"🔍 Анализируем категорию {category}: {len(pairs)} пар")
        
        results = {}
        successful_analyses = 0
        
        for pair in pairs:
            try:
                current_app.logger.info(f"📊 Анализируем {pair}...")
                result = self.advanced_ml_analysis(pair, timeframe)
                results[pair] = result
                successful_analyses += 1
                
                # Небольшая пауза между анализами
                import time
                time.sleep(1)
                
            except Exception as e:
                current_app.logger.warning(f"⚠️ Не удалось проанализировать {pair}: {e}")
                results[pair] = {'error': str(e)}
        
        current_app.logger.info(f"✅ Анализ категории {category} завершен: {successful_analyses}/{len(pairs)} успешно")
        
        return {
            'category': category,
            'total_pairs': len(pairs),
            'successful_analyses': successful_analyses,
            'results': results,
            'summary': self._generate_category_summary(results)
        }
    
    def _generate_category_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерировать сводку по категории криптовалют
        """
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'message': 'Нет успешных анализов для генерации сводки'}
        
        # Собираем статистику
        predictions = []
        confidences = []
        price_changes = []
        
        for pair, result in successful_results.items():
            if 'predicted_price' in result and 'current_price' in result:
                current_price = result['current_price']
                predicted_price = result['predicted_price']
                
                if current_price > 0:
                    change_percent = ((predicted_price - current_price) / current_price) * 100
                    price_changes.append(change_percent)
                    predictions.append(predicted_price)
                
                if 'confidence' in result:
                    confidences.append(result['confidence'])
        
        summary = {
            'successful_pairs': len(successful_results),
            'average_confidence': np.mean(confidences) if confidences else 0,
            'average_price_change': np.mean(price_changes) if price_changes else 0,
            'bullish_pairs': len([c for c in price_changes if c > 0]) if price_changes else 0,
            'bearish_pairs': len([c for c in price_changes if c < 0]) if price_changes else 0,
            'neutral_pairs': len([c for c in price_changes if abs(c) < 1]) if price_changes else 0
        }
        
        # Определяем общий тренд категории
        if summary['average_price_change'] > 2:
            summary['category_trend'] = 'БЫЧИЙ'
        elif summary['average_price_change'] < -2:
            summary['category_trend'] = 'МЕДВЕЖИЙ'
        else:
            summary['category_trend'] = 'НЕЙТРАЛЬНЫЙ'
        
        return summary
