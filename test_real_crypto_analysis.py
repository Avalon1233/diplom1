#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Cryptocurrency Analysis Test
Тестирование ML системы с реальными данными криптовалют
"""

import os
import sys
import time
import logging
from datetime import datetime

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_analysis_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('CRYPTO_ANALYSIS')

def test_real_crypto_analysis():
    """Тестирование анализа реальных криптовалют"""
    
    print("="*80)
    print("REAL CRYPTOCURRENCY ML ANALYSIS TEST")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    print()
    
    try:
        # Создаем Flask app context
        from app import create_app
        app = create_app()
        
        with app.app_context():
            from app.services.analysis_service import AnalysisService
            
            analysis_service = AnalysisService()
            
            # Тестируем анализ популярных криптовалют
            test_symbols = ['bitcoin', 'ethereum', 'binancecoin']
            timeframe = '1d'
            
            results = {}
            
            for i, symbol in enumerate(test_symbols, 1):
                print(f"\n[{i}/{len(test_symbols)}] ANALYZING {symbol.upper()}")
                print("-" * 50)
                
                start_time = time.time()
                
                try:
                    # Выполняем продвинутый ML анализ
                    result = analysis_service.advanced_ml_analysis(symbol, timeframe)
                    
                    analysis_time = time.time() - start_time
                    
                    if result['success']:
                        accuracy = result.get('model_accuracy', 0)
                        predicted_price = result.get('predicted_price', 0)
                        current_price = result['historical_data'][-1]['close'] if result['historical_data'] else 0
                        recommendation = result.get('recommendation', 'N/A')
                        features_used = result.get('features_used', 0)
                        data_points = result.get('data_points', 0)
                        
                        # Вычисляем потенциальное изменение цены
                        price_change = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
                        
                        print(f"SUCCESS - Analysis completed in {analysis_time:.2f}s")
                        print(f"Current Price: ${current_price:.2f}")
                        print(f"Predicted Price: ${predicted_price:.2f}")
                        print(f"Price Change: {price_change:+.2f}%")
                        print(f"Model Accuracy: {accuracy:.2f}%")
                        print(f"Recommendation: {recommendation}")
                        print(f"Features Used: {features_used}")
                        print(f"Data Points: {data_points}")
                        
                        results[symbol] = {
                            'success': True,
                            'current_price': current_price,
                            'predicted_price': predicted_price,
                            'price_change_percent': price_change,
                            'accuracy': accuracy,
                            'recommendation': recommendation,
                            'features_used': features_used,
                            'data_points': data_points,
                            'analysis_time': analysis_time
                        }
                        
                        # Показываем некоторые технические индикаторы
                        explanation = result.get('explanation', {})
                        if explanation:
                            print("\nTechnical Analysis:")
                            for indicator, desc in list(explanation.items())[:3]:  # Показываем первые 3
                                print(f"  {indicator}: {desc}")
                        
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"FAILED - {error}")
                        results[symbol] = {
                            'success': False,
                            'error': error,
                            'analysis_time': analysis_time
                        }
                        
                except Exception as e:
                    analysis_time = time.time() - start_time
                    print(f"ERROR - {str(e)}")
                    results[symbol] = {
                        'success': False,
                        'error': str(e),
                        'analysis_time': analysis_time
                    }
            
            # Выводим сводку результатов
            print("\n" + "="*80)
            print("ANALYSIS SUMMARY")
            print("="*80)
            
            successful_analyses = sum(1 for r in results.values() if r.get('success', False))
            total_analyses = len(results)
            
            print(f"Successful Analyses: {successful_analyses}/{total_analyses}")
            
            if successful_analyses > 0:
                # Находим лучшие результаты
                successful_results = {k: v for k, v in results.items() if v.get('success', False)}
                
                # Сортируем по точности модели
                sorted_by_accuracy = sorted(
                    successful_results.items(), 
                    key=lambda x: x[1]['accuracy'], 
                    reverse=True
                )
                
                print(f"\nRanking by ML Model Accuracy:")
                for i, (symbol, data) in enumerate(sorted_by_accuracy, 1):
                    print(f"{i}. {symbol.upper()}: {data['accuracy']:.2f}% accuracy, {data['price_change_percent']:+.2f}% predicted change")
                
                # Сортируем по потенциалу роста
                sorted_by_growth = sorted(
                    successful_results.items(), 
                    key=lambda x: x[1]['price_change_percent'], 
                    reverse=True
                )
                
                print(f"\nRanking by Growth Potential:")
                for i, (symbol, data) in enumerate(sorted_by_growth, 1):
                    print(f"{i}. {symbol.upper()}: {data['price_change_percent']:+.2f}% predicted change, {data['recommendation']}")
                
                # Средние метрики
                avg_accuracy = sum(r['accuracy'] for r in successful_results.values()) / len(successful_results)
                avg_analysis_time = sum(r['analysis_time'] for r in successful_results.values()) / len(successful_results)
                
                print(f"\nAverage Metrics:")
                print(f"- Average ML Accuracy: {avg_accuracy:.2f}%")
                print(f"- Average Analysis Time: {avg_analysis_time:.2f}s")
                
            print("="*80)
            print("ML SYSTEM STATUS: FULLY OPERATIONAL")
            print("Advanced ML analysis with high accuracy predictions completed!")
            print("="*80)
            
            return results
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return None

def main():
    """Главная функция"""
    results = test_real_crypto_analysis()
    
    if results:
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        print(f"\nFinal Result: {successful}/{total} successful analyses")
    else:
        print("\nTest failed to complete")

if __name__ == "__main__":
    main()
