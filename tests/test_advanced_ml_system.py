# tests/test_advanced_ml_system.py
"""
Комплексный тест для новой продвинутой ML системы анализа
"""
import pytest
import pandas as pd
from app import create_app
from app.services.analysis_service import AnalysisService


@pytest.fixture(scope='module')
def test_client():
    """Создает тестовый клиент Flask"""
    flask_app = create_app('testing')

    with flask_app.test_client() as testing_client:
        with flask_app.app_context():
            yield testing_client


def test_analysis_service_initialization(test_client):
    """Тест: Инициализация AnalysisService"""
    try:
        service = AnalysisService()
        assert service is not None, "AnalysisService не должен быть None"
        assert service.feature_engineer is not None, "AdvancedFeatureEngineering не инициализирован"
        assert service.ensemble_model is not None, "EnsembleMLModel не инициализирован"
    except Exception as e:
        pytest.fail(f"Ошибка инициализации AnalysisService: {e}")


@pytest.mark.slow
def test_advanced_ml_analysis_btc(test_client):
    """Тест: Продвинутый ML анализ для BTC-USD"""
    service = AnalysisService()
    symbol = 'BTC-USD'
    timeframe = '1d'

    # Выполняем анализ
    result = service.advanced_ml_analysis(symbol, timeframe)

    # Проверяем успешность анализа
    assert result['success'], f"Ошибка анализа: {result.get('error', 'Неизвестная ошибка')}"

    # Проверяем наличие всех ключевых полей
    required_keys = [
        'symbol', 'timeframe', 'historical_data', 'predicted_price',
        'confidence_interval', 'uncertainty', 'recommendation', 'explanation',
        'model_accuracy', 'training_metrics', 'prediction_details',
        'features_used', 'data_points'
    ]
    for key in required_keys:
        assert key in result, f"Отсутствует ключ '{key}' в результатах анализа"

    # Проверяем типы данных
    assert result['symbol'] == symbol
    assert isinstance(result['predicted_price'], float)
    assert isinstance(result['model_accuracy'], float)
    assert isinstance(result['recommendation'], str)
    assert isinstance(result['explanation'], dict)
    assert isinstance(result['historical_data'], list)

    # Проверяем значения
    assert result['model_accuracy'] > 0, "Точность модели должна быть положительной"
    assert result['predicted_price'] > 0, "Предсказанная цена должна быть положительной"
    assert len(result['confidence_interval']) == 2, "Доверительный интервал должен состоять из 2 значений"
    assert result['confidence_interval'][0] < result['confidence_interval'][1], "Нижняя граница интервала должна быть меньше верхней"
    assert len(result['explanation']) > 5, "Должно быть как минимум 5 объяснений (XAI)"

    print(f"\n✅ Тест для {symbol} пройден успешно!")
    print(f"  - Предсказанная цена: ${result['predicted_price']:.2f}")
    print(f"  - Точность модели: {result['model_accuracy']:.2f}%")
    print(f"  - Рекомендация: {result['recommendation']}")


@pytest.mark.slow
def test_compare_cryptocurrencies(test_client):
    """Тест: Сравнение нескольких криптовалют"""
    service = AnalysisService()
    symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']
    timeframe = '1d'

    result = service.compare_cryptocurrencies(symbols, timeframe, 'ml_prediction')

    assert result['success'], f"Ошибка сравнения: {result.get('error', 'Неизвестная ошибка')}"
    assert 'comparison_results' in result
    assert len(result['comparison_results']) > 0, "Результаты сравнения не должны быть пустыми"

    for symbol in result['comparison_results']:
        assert 'accuracy' in result['comparison_results'][symbol]
        assert 'recommendation' in result['comparison_results'][symbol]

    print(f"\n✅ Тест сравнения для {', '.join(symbols)} пройден успешно!")

