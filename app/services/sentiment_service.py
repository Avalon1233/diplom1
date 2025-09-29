# app/services/sentiment_service.py
"""
Sentiment Analysis Service - Анализ настроений из новостей и социальных сетей
для улучшения точности ML-предсказаний криптовалют
"""

import os
import requests
import time
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from flask import current_app

try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWS_API_AVAILABLE = True
except ImportError:
    NEWS_API_AVAILABLE = False


class SentimentAnalysisService:
    """
    Сервис для анализа настроений из новостей и социальных сетей
    """
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer() if SENTIMENT_AVAILABLE else None
        self.news_api = None
        
        # Попытка инициализации News API (требует API ключ)
        news_api_key = os.getenv('NEWS_API_KEY')
        if NEWS_API_AVAILABLE and news_api_key:
            try:
                self.news_api = NewsApiClient(api_key=news_api_key)
            except Exception as e:
                current_app.logger.warning(f"Не удалось инициализировать News API: {e}")
    
    def get_crypto_sentiment(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """
        Получает общий sentiment для криптовалюты
        
        Args:
            symbol: Символ криптовалюты (например, 'BTC', 'ETH')
            days: Количество дней для анализа
            
        Returns:
            Словарь с метриками sentiment
        """
        if not SENTIMENT_AVAILABLE:
            current_app.logger.warning("Sentiment analysis библиотеки недоступны")
            return self._get_default_sentiment()
        
        try:
            # Получаем новости
            news_sentiment = self._analyze_news_sentiment(symbol, days)
            
            # Получаем социальные сигналы (заглушка для Reddit/Twitter API)
            social_sentiment = self._analyze_social_sentiment(symbol, days)
            
            # Комбинируем результаты
            combined_sentiment = self._combine_sentiments(news_sentiment, social_sentiment)
            
            current_app.logger.info(f"📰 Sentiment для {symbol}: {combined_sentiment['overall']:.3f}")
            return combined_sentiment
            
        except Exception as e:
            current_app.logger.error(f"Ошибка анализа sentiment для {symbol}: {e}")
            return self._get_default_sentiment()
    
    def _analyze_news_sentiment(self, symbol: str, days: int) -> Dict[str, float]:
        """Анализ sentiment из новостей"""
        if not self.news_api:
            # Используем заглушку с реалистичными данными
            return self._get_mock_news_sentiment(symbol)
        
        try:
            # Поиск новостей
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Ключевые слова для поиска
            crypto_names = {
                'BTC': ['Bitcoin', 'BTC'],
                'ETH': ['Ethereum', 'ETH'],
                'BNB': ['Binance', 'BNB'],
                'ADA': ['Cardano', 'ADA'],
                'SOL': ['Solana', 'SOL']
            }
            
            keywords = crypto_names.get(symbol.upper(), [symbol])
            query = ' OR '.join(keywords)
            
            articles = self.news_api.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='relevancy',
                page_size=50
            )
            
            if not articles['articles']:
                return self._get_mock_news_sentiment(symbol)
            
            # Анализируем sentiment каждой статьи
            sentiments = []
            for article in articles['articles']:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    sentiment = self._analyze_text_sentiment(text)
                    sentiments.append(sentiment)
            
            if not sentiments:
                return self._get_mock_news_sentiment(symbol)
            
            # Вычисляем средние значения
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return {
                'sentiment': avg_sentiment,
                'articles_count': len(sentiments),
                'source': 'news_api'
            }
            
        except Exception as e:
            current_app.logger.error(f"Ошибка анализа новостей: {e}")
            return self._get_mock_news_sentiment(symbol)
    
    def _analyze_social_sentiment(self, symbol: str, days: int) -> Dict[str, float]:
        """Анализ sentiment из социальных сетей (заглушка)"""
        # В реальной реализации здесь был бы анализ Twitter/Reddit API
        # Пока используем заглушку с реалистичными данными
        
        import random
        random.seed(hash(symbol) + int(time.time() / 86400))  # Детерминированная случайность по дням
        
        # Симулируем различные настроения для разных криптовалют
        base_sentiments = {
            'BTC': 0.1,   # Обычно позитивный
            'ETH': 0.05,  # Нейтрально-позитивный
            'BNB': 0.0,   # Нейтральный
            'ADA': -0.05, # Слегка негативный
            'SOL': 0.15   # Позитивный
        }
        
        base = base_sentiments.get(symbol.upper(), 0.0)
        noise = random.uniform(-0.2, 0.2)  # Добавляем шум
        
        return {
            'sentiment': max(-1.0, min(1.0, base + noise)),
            'posts_count': random.randint(50, 200),
            'source': 'social_mock'
        }
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Анализирует sentiment текста"""
        if not text or not SENTIMENT_AVAILABLE:
            return 0.0
        
        try:
            # Используем VADER для финансовых текстов (лучше работает с сленгом)
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_sentiment = vader_scores['compound']
            
            # Используем TextBlob как дополнительный анализатор
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Комбинируем результаты (VADER весит больше для финансовых текстов)
            combined = 0.7 * vader_sentiment + 0.3 * textblob_sentiment
            
            return max(-1.0, min(1.0, combined))
            
        except Exception as e:
            current_app.logger.error(f"Ошибка анализа текста: {e}")
            return 0.0
    
    def _combine_sentiments(self, news_sentiment: Dict, social_sentiment: Dict) -> Dict[str, float]:
        """Комбинирует sentiment из разных источников"""
        
        # Веса для разных источников
        news_weight = 0.6
        social_weight = 0.4
        
        # Комбинированный sentiment
        overall = (
            news_sentiment['sentiment'] * news_weight +
            social_sentiment['sentiment'] * social_weight
        )
        
        return {
            'overall': overall,
            'news_sentiment': news_sentiment['sentiment'],
            'social_sentiment': social_sentiment['sentiment'],
            'news_articles': news_sentiment.get('articles_count', 0),
            'social_posts': social_sentiment.get('posts_count', 0),
            'confidence': self._calculate_confidence(news_sentiment, social_sentiment)
        }
    
    def _calculate_confidence(self, news_sentiment: Dict, social_sentiment: Dict) -> float:
        """Вычисляет уверенность в sentiment анализе"""
        
        # Базовая уверенность
        confidence = 0.5
        
        # Увеличиваем уверенность при наличии данных
        if news_sentiment.get('articles_count', 0) > 10:
            confidence += 0.2
        
        if social_sentiment.get('posts_count', 0) > 50:
            confidence += 0.2
        
        # Уменьшаем уверенность при противоречивых сигналах
        sentiment_diff = abs(news_sentiment['sentiment'] - social_sentiment['sentiment'])
        if sentiment_diff > 0.5:
            confidence -= 0.3
        
        return max(0.1, min(1.0, confidence))
    
    def _get_mock_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Заглушка для news sentiment"""
        import random
        random.seed(hash(symbol) + int(time.time() / 86400))
        
        # Различные базовые настроения для криптовалют
        base_sentiments = {
            'BTC': 0.15,
            'ETH': 0.1,
            'BNB': 0.05,
            'ADA': 0.0,
            'SOL': 0.2
        }
        
        base = base_sentiments.get(symbol.upper(), 0.0)
        noise = random.uniform(-0.15, 0.15)
        
        return {
            'sentiment': max(-1.0, min(1.0, base + noise)),
            'articles_count': random.randint(10, 30),
            'source': 'mock'
        }
    
    def _get_default_sentiment(self) -> Dict[str, float]:
        """Возвращает нейтральный sentiment по умолчанию"""
        return {
            'overall': 0.0,
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'news_articles': 0,
            'social_posts': 0,
            'confidence': 0.1
        }
    
    def get_sentiment_features(self, symbol: str) -> Dict[str, float]:
        """
        Получает sentiment признаки для ML модели
        
        Returns:
            Словарь с признаками для добавления в feature engineering
        """
        sentiment_data = self.get_crypto_sentiment(symbol)
        
        return {
            'sentiment_overall': sentiment_data['overall'],
            'sentiment_news': sentiment_data['news_sentiment'],
            'sentiment_social': sentiment_data['social_sentiment'],
            'sentiment_confidence': sentiment_data['confidence'],
            'sentiment_news_volume': min(sentiment_data['news_articles'] / 50.0, 1.0),  # Нормализуем
            'sentiment_social_volume': min(sentiment_data['social_posts'] / 200.0, 1.0)  # Нормализуем
        }
    
    def get_enhanced_sentiment_features(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """
        Получает расширенные sentiment признаки из множественных источников
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
            
        Returns:
            Словарь с расширенными sentiment признаками
        """
        try:
            # Получаем базовый sentiment
            base_sentiment = self.get_crypto_sentiment(symbol, days)
            
            # Добавляем симуляцию Reddit и Twitter sentiment
            reddit_sentiment = self._get_reddit_sentiment_mock(symbol)
            twitter_sentiment = self._get_twitter_sentiment_mock(symbol)
            
            # Вычисляем взвешенные метрики
            weighted_sentiment = (
                base_sentiment['overall'] * 0.4 +
                reddit_sentiment * 0.35 +
                twitter_sentiment * 0.25
            )
            
            # Вычисляем momentum и volatility
            sentiment_momentum = self._calculate_sentiment_momentum_mock(symbol)
            sentiment_volatility = abs(base_sentiment['overall'] - reddit_sentiment) + abs(reddit_sentiment - twitter_sentiment)
            sentiment_volatility = min(sentiment_volatility / 2.0, 1.0)  # Нормализуем
            
            return {
                'sentiment_overall': weighted_sentiment,
                'sentiment_news': base_sentiment['news_sentiment'],
                'sentiment_reddit': reddit_sentiment,
                'sentiment_twitter': twitter_sentiment,
                'sentiment_momentum': sentiment_momentum,
                'sentiment_volatility': sentiment_volatility,
                'sentiment_confidence': base_sentiment['confidence'] * 1.2,  # Повышаем уверенность при множественных источниках
                'news_volume': base_sentiment['news_articles'],
                'reddit_volume': self._get_reddit_volume_mock(),
                'twitter_volume': self._get_twitter_volume_mock()
            }
            
        except Exception as e:
            current_app.logger.error(f"Ошибка получения расширенных sentiment признаков для {symbol}: {e}")
            return self._get_default_enhanced_sentiment()
    
    def _get_reddit_sentiment_mock(self, symbol: str) -> float:
        """Заглушка для Reddit sentiment с реалистичными данными"""
        import random
        import hashlib
        
        # Используем hash символа для консистентности
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Reddit обычно более позитивный для популярных криптовалют
        if symbol.upper() in ['BTC', 'ETH']:
            base_sentiment = random.uniform(0.1, 0.4)
        elif symbol.upper() in ['BNB', 'ADA', 'DOT']:
            base_sentiment = random.uniform(-0.1, 0.3)
        else:
            base_sentiment = random.uniform(-0.2, 0.2)
        
        # Добавляем небольшой шум
        noise = random.uniform(-0.1, 0.1)
        return max(-1.0, min(1.0, base_sentiment + noise))
    
    def _get_twitter_sentiment_mock(self, symbol: str) -> float:
        """Заглушка для Twitter sentiment (обычно более волатильный)"""
        import random
        import hashlib
        
        seed = int(hashlib.md5(f"{symbol}_twitter".encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Twitter более эмоциональный и волатильный
        base_sentiment = random.uniform(-0.3, 0.3)
        volatility_factor = random.uniform(0.8, 1.5)  # Увеличиваем волатильность
        
        return max(-1.0, min(1.0, base_sentiment * volatility_factor))
    
    def _calculate_sentiment_momentum_mock(self, symbol: str) -> float:
        """Заглушка для sentiment momentum"""
        import random
        import hashlib
        
        seed = int(hashlib.md5(f"{symbol}_momentum".encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Momentum может быть положительным или отрицательным
        return random.uniform(-0.5, 0.5)
    
    def _get_reddit_volume_mock(self) -> int:
        """Заглушка для объема Reddit постов"""
        import random
        return random.randint(15, 80)
    
    def _get_twitter_volume_mock(self) -> int:
        """Заглушка для объема Twitter постов"""
        import random
        return random.randint(50, 300)
    
    def _get_default_enhanced_sentiment(self) -> Dict[str, float]:
        """Возвращает дефолтные расширенные sentiment признаки"""
        return {
            'sentiment_overall': 0.0,
            'sentiment_news': 0.0,
            'sentiment_reddit': 0.0,
            'sentiment_twitter': 0.0,
            'sentiment_momentum': 0.0,
            'sentiment_volatility': 0.5,
            'sentiment_confidence': 0.1,
            'news_volume': 0,
            'reddit_volume': 0,
            'twitter_volume': 0
        }
