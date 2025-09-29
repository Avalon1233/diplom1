# app/services/macro_indicators_service.py
"""
Макроэкономические индикаторы для улучшения точности ML-предсказаний криптовалют
Включает индексы страха и жадности, корреляции с традиционными активами, и другие макро-факторы
"""

import os
import requests
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from flask import current_app
import numpy as np


class MacroIndicatorsService:
    """
    Сервис для получения и анализа макроэкономических индикаторов
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 час кэширования
        
    def get_macro_features(self, symbol: str) -> Dict[str, float]:
        """
        Получает макроэкономические признаки для криптовалюты
        
        Args:
            symbol: Символ криптовалюты
            
        Returns:
            Словарь с макроэкономическими признаками
        """
        try:
            # Проверяем кэш
            cache_key = f"macro_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Получаем различные макроэкономические индикаторы
            fear_greed = self._get_fear_greed_index()
            market_dominance = self._get_market_dominance(symbol)
            correlation_metrics = self._get_correlation_metrics(symbol)
            volatility_metrics = self._get_volatility_metrics(symbol)
            liquidity_metrics = self._get_liquidity_metrics(symbol)
            
            # Объединяем все признаки
            macro_features = {
                **fear_greed,
                **market_dominance,
                **correlation_metrics,
                **volatility_metrics,
                **liquidity_metrics
            }
            
            # Кэшируем результат
            self.cache[cache_key] = {
                'data': macro_features,
                'timestamp': time.time()
            }
            
            current_app.logger.info(f"📊 Получено {len(macro_features)} макроэкономических признаков для {symbol}")
            return macro_features
            
        except Exception as e:
            current_app.logger.error(f"Ошибка получения макроэкономических признаков для {symbol}: {e}")
            return self._get_default_macro_features()
    
    def _get_fear_greed_index(self) -> Dict[str, float]:
        """
        Получает индекс страха и жадности (Fear & Greed Index)
        В реальной реализации использовать API alternative.me
        """
        try:
            # Симуляция Fear & Greed Index (0-100)
            import random
            import hashlib
            
            # Используем текущую дату для консистентности в течение дня
            today = datetime.now().strftime('%Y-%m-%d')
            seed = int(hashlib.md5(today.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # Fear & Greed обычно колеблется между 10-90
            current_fgi = random.randint(15, 85)
            
            # Симулируем исторические значения для momentum
            yesterday_fgi = current_fgi + random.randint(-15, 15)
            week_ago_fgi = current_fgi + random.randint(-25, 25)
            
            # Нормализуем значения (0-100 -> 0-1)
            return {
                'fear_greed_index': current_fgi / 100.0,
                'fear_greed_momentum_1d': (current_fgi - yesterday_fgi) / 100.0,
                'fear_greed_momentum_7d': (current_fgi - week_ago_fgi) / 100.0,
                'fear_greed_volatility': abs(current_fgi - yesterday_fgi) / 100.0
            }
            
        except Exception as e:
            current_app.logger.warning(f"Ошибка получения Fear & Greed Index: {e}")
            return {
                'fear_greed_index': 0.5,
                'fear_greed_momentum_1d': 0.0,
                'fear_greed_momentum_7d': 0.0,
                'fear_greed_volatility': 0.1
            }
    
    def _get_market_dominance(self, symbol: str) -> Dict[str, float]:
        """
        Получает метрики доминирования рынка
        """
        try:
            import random
            import hashlib
            
            seed = int(hashlib.md5(f"{symbol}_dominance".encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # Симулируем доминирование BTC (обычно 40-70%)
            btc_dominance = random.uniform(0.4, 0.7)
            
            # Доминирование ETH (обычно 15-25%)
            eth_dominance = random.uniform(0.15, 0.25)
            
            # Доминирование топ-10 (обычно 80-95%)
            top10_dominance = random.uniform(0.8, 0.95)
            
            # Специфичные метрики для символа
            if symbol.upper() == 'BTC':
                symbol_dominance = btc_dominance
            elif symbol.upper() == 'ETH':
                symbol_dominance = eth_dominance
            else:
                symbol_dominance = random.uniform(0.001, 0.05)  # Альткоины
            
            return {
                'btc_dominance': btc_dominance,
                'eth_dominance': eth_dominance,
                'top10_dominance': top10_dominance,
                'symbol_dominance': symbol_dominance,
                'altcoin_season_indicator': 1.0 - btc_dominance  # Обратная корреляция
            }
            
        except Exception as e:
            current_app.logger.warning(f"Ошибка получения market dominance для {symbol}: {e}")
            return {
                'btc_dominance': 0.5,
                'eth_dominance': 0.2,
                'top10_dominance': 0.85,
                'symbol_dominance': 0.01,
                'altcoin_season_indicator': 0.5
            }
    
    def _get_correlation_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Получает корреляции с традиционными активами
        """
        try:
            import random
            import hashlib
            
            seed = int(hashlib.md5(f"{symbol}_correlation".encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # Симулируем корреляции с различными активами
            # Криптовалюты обычно имеют низкую корреляцию с традиционными активами
            
            correlations = {
                'correlation_sp500': random.uniform(-0.3, 0.4),      # S&P 500
                'correlation_gold': random.uniform(-0.2, 0.3),       # Золото
                'correlation_dxy': random.uniform(-0.4, 0.2),        # Доллар США
                'correlation_vix': random.uniform(-0.1, 0.5),        # VIX (волатильность)
                'correlation_bonds': random.uniform(-0.3, 0.1),      # Облигации
                'correlation_oil': random.uniform(-0.2, 0.3),        # Нефть
                'correlation_btc': 1.0 if symbol.upper() == 'BTC' else random.uniform(0.6, 0.9)  # Корреляция с BTC
            }
            
            # Добавляем метрики стабильности корреляций
            correlations['correlation_stability'] = 1.0 - np.std(list(correlations.values()))
            correlations['max_correlation'] = max(abs(v) for v in correlations.values() if v != 1.0)
            
            return correlations
            
        except Exception as e:
            current_app.logger.warning(f"Ошибка получения correlation metrics для {symbol}: {e}")
            return {
                'correlation_sp500': 0.0,
                'correlation_gold': 0.0,
                'correlation_dxy': 0.0,
                'correlation_vix': 0.0,
                'correlation_bonds': 0.0,
                'correlation_oil': 0.0,
                'correlation_btc': 0.8,
                'correlation_stability': 0.5,
                'max_correlation': 0.3
            }
    
    def _get_volatility_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Получает метрики волатильности рынка
        """
        try:
            import random
            import hashlib
            
            seed = int(hashlib.md5(f"{symbol}_volatility".encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # Симулируем различные метрики волатильности
            base_volatility = random.uniform(0.02, 0.08)  # 2-8% дневная волатильность
            
            return {
                'market_volatility_1d': base_volatility,
                'market_volatility_7d': base_volatility * random.uniform(0.8, 1.2),
                'market_volatility_30d': base_volatility * random.uniform(0.7, 1.3),
                'volatility_trend': random.uniform(-0.5, 0.5),  # Тренд волатильности
                'volatility_percentile': random.uniform(0.1, 0.9),  # Перцентиль исторической волатильности
                'implied_volatility': base_volatility * random.uniform(1.1, 1.5)  # Подразумеваемая волатильность
            }
            
        except Exception as e:
            current_app.logger.warning(f"Ошибка получения volatility metrics для {symbol}: {e}")
            return {
                'market_volatility_1d': 0.04,
                'market_volatility_7d': 0.04,
                'market_volatility_30d': 0.04,
                'volatility_trend': 0.0,
                'volatility_percentile': 0.5,
                'implied_volatility': 0.05
            }
    
    def _get_liquidity_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Получает метрики ликвидности рынка
        """
        try:
            import random
            import hashlib
            
            seed = int(hashlib.md5(f"{symbol}_liquidity".encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # Симулируем метрики ликвидности
            # Более популярные криптовалюты имеют лучшую ликвидность
            if symbol.upper() in ['BTC', 'ETH']:
                base_liquidity = random.uniform(0.7, 0.95)
            elif symbol.upper() in ['BNB', 'ADA', 'DOT']:
                base_liquidity = random.uniform(0.5, 0.8)
            else:
                base_liquidity = random.uniform(0.2, 0.6)
            
            return {
                'market_liquidity': base_liquidity,
                'bid_ask_spread': (1.0 - base_liquidity) * 0.01,  # Обратная корреляция
                'order_book_depth': base_liquidity,
                'trading_volume_ratio': base_liquidity * random.uniform(0.8, 1.2),
                'liquidity_trend': random.uniform(-0.2, 0.2),
                'exchange_flow_ratio': random.uniform(0.3, 0.7)  # Отношение притока/оттока с бирж
            }
            
        except Exception as e:
            current_app.logger.warning(f"Ошибка получения liquidity metrics для {symbol}: {e}")
            return {
                'market_liquidity': 0.5,
                'bid_ask_spread': 0.005,
                'order_book_depth': 0.5,
                'trading_volume_ratio': 0.5,
                'liquidity_trend': 0.0,
                'exchange_flow_ratio': 0.5
            }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Проверяет валидность кэша"""
        if cache_key not in self.cache:
            return False
        
        cache_age = time.time() - self.cache[cache_key]['timestamp']
        return cache_age < self.cache_ttl
    
    def _get_default_macro_features(self) -> Dict[str, float]:
        """Возвращает дефолтные макроэкономические признаки"""
        return {
            # Fear & Greed Index
            'fear_greed_index': 0.5,
            'fear_greed_momentum_1d': 0.0,
            'fear_greed_momentum_7d': 0.0,
            'fear_greed_volatility': 0.1,
            
            # Market Dominance
            'btc_dominance': 0.5,
            'eth_dominance': 0.2,
            'top10_dominance': 0.85,
            'symbol_dominance': 0.01,
            'altcoin_season_indicator': 0.5,
            
            # Correlations
            'correlation_sp500': 0.0,
            'correlation_gold': 0.0,
            'correlation_dxy': 0.0,
            'correlation_vix': 0.0,
            'correlation_bonds': 0.0,
            'correlation_oil': 0.0,
            'correlation_btc': 0.8,
            'correlation_stability': 0.5,
            'max_correlation': 0.3,
            
            # Volatility
            'market_volatility_1d': 0.04,
            'market_volatility_7d': 0.04,
            'market_volatility_30d': 0.04,
            'volatility_trend': 0.0,
            'volatility_percentile': 0.5,
            'implied_volatility': 0.05,
            
            # Liquidity
            'market_liquidity': 0.5,
            'bid_ask_spread': 0.005,
            'order_book_depth': 0.5,
            'trading_volume_ratio': 0.5,
            'liquidity_trend': 0.0,
            'exchange_flow_ratio': 0.5
        }
    
    def get_enhanced_macro_features(self, symbol: str, market_cap_rank: int = None) -> Dict[str, float]:
        """
        Получает расширенные макроэкономические признаки с учетом ранга по капитализации
        
        Args:
            symbol: Символ криптовалюты
            market_cap_rank: Ранг по рыночной капитализации
            
        Returns:
            Словарь с расширенными макроэкономическими признаками
        """
        try:
            # Получаем базовые макро признаки
            base_features = self.get_macro_features(symbol)
            
            # Добавляем признаки на основе ранга капитализации
            if market_cap_rank:
                rank_features = self._get_market_cap_rank_features(market_cap_rank)
                base_features.update(rank_features)
            
            # Добавляем сезонные признаки
            seasonal_features = self._get_seasonal_features()
            base_features.update(seasonal_features)
            
            # Добавляем макроэкономические циклы
            cycle_features = self._get_economic_cycle_features()
            base_features.update(cycle_features)
            
            return base_features
            
        except Exception as e:
            current_app.logger.error(f"Ошибка получения расширенных макро признаков для {symbol}: {e}")
            return self._get_default_macro_features()
    
    def _get_market_cap_rank_features(self, rank: int) -> Dict[str, float]:
        """Получает признаки на основе ранга по капитализации"""
        return {
            'market_cap_rank': min(rank / 100.0, 1.0),  # Нормализуем
            'is_top10': 1.0 if rank <= 10 else 0.0,
            'is_top50': 1.0 if rank <= 50 else 0.0,
            'is_top100': 1.0 if rank <= 100 else 0.0,
            'rank_stability': max(0.0, 1.0 - rank / 1000.0)  # Стабильность ранга
        }
    
    def _get_seasonal_features(self) -> Dict[str, float]:
        """Получает сезонные признаки"""
        now = datetime.now()
        
        return {
            'month_sin': np.sin(2 * np.pi * now.month / 12),
            'month_cos': np.cos(2 * np.pi * now.month / 12),
            'day_of_week_sin': np.sin(2 * np.pi * now.weekday() / 7),
            'day_of_week_cos': np.cos(2 * np.pi * now.weekday() / 7),
            'is_weekend': 1.0 if now.weekday() >= 5 else 0.0,
            'is_month_end': 1.0 if now.day >= 28 else 0.0
        }
    
    def _get_economic_cycle_features(self) -> Dict[str, float]:
        """Получает признаки экономических циклов"""
        import random
        
        return {
            'economic_cycle_phase': random.uniform(0.0, 1.0),  # 0=recession, 1=expansion
            'interest_rate_trend': random.uniform(-0.5, 0.5),
            'inflation_expectation': random.uniform(0.0, 0.1),
            'risk_appetite': random.uniform(0.2, 0.8),
            'monetary_policy_stance': random.uniform(-1.0, 1.0)  # -1=dovish, 1=hawkish
        }
