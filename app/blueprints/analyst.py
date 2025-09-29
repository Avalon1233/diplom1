# app/blueprints/analyst.py
"""
Analyst blueprint for cryptocurrency analysis, comparisons, and research tools
"""
from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from app.models import db, SystemMetrics
from app.forms import AnalysisForm, CompareForm
from app.utils.decorators import role_required, log_user_activity, measure_performance
from app.services.crypto_service import CryptoService
from app.services.analysis_service import AnalysisService
from app.utils.security import sanitize_input
from app.utils.validators import validate_crypto_symbol

analyst_bp = Blueprint('analyst', __name__, url_prefix='/analyst')


@analyst_bp.route('/dashboard')
@login_required
@role_required('analyst')
@log_user_activity
@measure_performance
def dashboard():
    """Analyst dashboard with market overview and analysis tools"""
    try:
        # Get recent analysis metrics
        recent_analyses = SystemMetrics.query.filter_by(
            metric_name='analysis_request'
        ).filter(
            SystemMetrics.tags.contains({'user_id': current_user.id})
        ).order_by(SystemMetrics.created_at.desc()).limit(10).all()

        # Get market overview
        crypto_service = CryptoService()
        try:
            market_overview = crypto_service.get_market_data([
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT'
            ])
        except Exception as e:
            current_app.logger.error(
                f"Failed to get market overview: {str(e)}")
            market_overview = []

        # Get trending cryptocurrencies
        try:
            trending_cryptos = crypto_service.get_market_data([
                'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT'
            ])
        except Exception as e:
            current_app.logger.error(
                f"Failed to get trending cryptos: {str(e)}")
            trending_cryptos = []

        # Analysis statistics
        analysis_stats = {
            'total_analyses': len(recent_analyses),
            'this_week': len([a for a in recent_analyses if a.created_at > datetime.utcnow() - timedelta(days=7)]),
            'favorite_crypto': 'BTC-USD',  # Default
            'accuracy_rate': 85.2  # Demo value
        }

        return render_template('analyst/dashboard.html',
                               recent_analyses=recent_analyses,
                               market_overview=market_overview,
                               trending_cryptos=trending_cryptos,
                               analysis_stats=analysis_stats)

    except Exception as e:
        current_app.logger.error(f"Analyst dashboard error: {str(e)}")
        return render_template('analyst/dashboard.html',
                               recent_analyses=[],
                               market_overview=[],
                               trending_cryptos=[],
                               analysis_stats={})


@analyst_bp.route('/analyze', methods=['GET', 'POST'])
@login_required
@role_required('analyst')
@log_user_activity
@measure_performance
def analyze():
    """Renders the analysis page with the necessary form."""
    analysis_form = AnalysisForm()
    # Populate choices dynamically
    crypto_service = CryptoService()
    available_symbols = crypto_service.get_supported_symbols()
    analysis_form.symbol.choices = [(s, s) for s in available_symbols]
    return render_template('analyst/analyze.html', form=analysis_form)


@analyst_bp.route('/compare', methods=['GET', 'POST'])
@login_required
@role_required('analyst')
@log_user_activity
@measure_performance
def compare():
    """Comparison page for cryptocurrency analysis"""
    try:
        comparison_form = CompareForm()

        # Get available cryptocurrencies for comparison
        available_cryptos = [
            ('BTC-USD', 'Bitcoin'),
            ('ETH-USD', 'Ethereum'),
            ('BNB-USD', 'Binance Coin'),
            ('ADA-USD', 'Cardano'),
            ('SOL-USD', 'Solana'),
            ('DOT-USD', 'Polkadot'),
            ('AVAX-USD', 'Avalanche'),
            ('MATIC-USD', 'Polygon'),
            ('LINK-USD', 'Chainlink'),
            ('UNI-USD', 'Uniswap')
        ]

        # Set choices for form
        comparison_form.symbols.choices = available_cryptos

        plot_div = None
        results = None

        if comparison_form.validate_on_submit():
            try:
                symbols = comparison_form.symbols.data
                timeframe = comparison_form.timeframe.data
                comparison_type = comparison_form.comparison_type.data

                # Perform comparison analysis
                analysis_service = AnalysisService()
                comparison_result = analysis_service.compare_cryptocurrencies(
                    symbols=symbols,
                    timeframe=timeframe,
                    comparison_type=comparison_type
                )

                plot_div = comparison_result.get('plot_html')
                results = comparison_result.get('summary_data', [])

            except Exception as e:
                current_app.logger.error(
                    f"Comparison analysis error: {str(e)}")
                flash('Ошибка при выполнении сравнения', 'error')

        return render_template('analyst/compare.html',
                               form=comparison_form,
                               available_cryptos=available_cryptos,
                               plot_div=plot_div,
                               results=results)

    except Exception as e:
        current_app.logger.error(f"Compare page error: {str(e)}")
        return render_template('analyst/compare.html',
                               comparison_form=CompareForm(),
                               available_cryptos=[])


@analyst_bp.route('/api/analyze', methods=['POST'])
@login_required
@role_required('analyst')
@log_user_activity
@measure_performance
def perform_analysis():
    """Perform cryptocurrency analysis using the appropriate service method."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        symbol = sanitize_input(data.get('symbol'))
        timeframe = sanitize_input(data.get('timeframe'))
        analysis_type = sanitize_input(data.get('analysis_type'))

        if not all([symbol, timeframe, analysis_type]):
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400

        analysis_service = AnalysisService()

        if analysis_type == 'neural':
            # Вызов нового продвинутого ML анализа
            result = analysis_service.advanced_ml_analysis(symbol, timeframe)
            
            # Адаптация ответа для фронтенда
            if result.get('status') == 'success':
                # Конвертируем NumPy типы в стандартные Python типы для JSON сериализации
                def convert_numpy_types(obj):
                    import numpy as np
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(v) for v in obj]
                    return obj
                
                # Конвертируем все данные
                result = convert_numpy_types(result)
                
                result['success'] = True
                # Фронтенд ожидает эти поля, добавим заглушки, если их нет
                if 'historical_data' not in result:
                    # Создаем минимальные исторические данные для графика
                    import datetime
                    current_price = result.get('current_price', 50000)
                    result['historical_data'] = [
                        {
                            'timestamp': (datetime.datetime.now() - datetime.timedelta(days=i)).isoformat(),
                            'close': current_price * (1 + (i * 0.001))  # Небольшие вариации цены
                        } for i in range(30, 0, -1)  # 30 дней истории
                    ]
                if 'model_accuracy' not in result:
                    result['model_accuracy'] = result.get('training_metrics', {}).get('ensemble', {}).get('accuracy', 0)
                if 'confidence_interval' not in result:
                    pred_price = result.get('predicted_price', 0)
                    std_dev = result.get('prediction_std_dev', 0)
                    result['confidence_interval'] = [pred_price - 1.96 * std_dev, pred_price + 1.96 * std_dev]
                if 'uncertainty' not in result:
                    result['uncertainty'] = result.get('prediction_std_dev', 0)
                if 'explanation' not in result:
                    result['explanation'] = result.get('insights', {})

                return jsonify(result)
            else:
                return jsonify({'success': False, 'error': result.get('message', 'Произошла ошибка при анализе.')}), 500
        else:
            # Заглушка для старых типов анализа, чтобы не ломать интерфейс
            # В будущем их можно будет также перевести на новый сервис
            return jsonify({
                'success': True,
                'analysis_type': analysis_type,
                'message': f'{analysis_type.capitalize()} analysis is not yet migrated to the new service.'
            })

    except Exception as e:
        current_app.logger.error(f"Analysis API error: {str(e)}")
        return jsonify({'success': False, 'error': 'An internal error occurred'}), 500


# @analyst_bp.route('/api/compare', methods=['POST'])
# @login_required
# @role_required('analyst')
# @log_user_activity
# @measure_performance
# def perform_comparison():
#     """Perform cryptocurrency comparison"""
#     # This endpoint needs to be refactored to work with the new service architecture.
#     # Temporarily disabled to prevent errors.
#     return jsonify({'success': False, 'error': 'This feature is temporarily disabled.'}), 503


@analyst_bp.route('/api/market_data/<symbol>')
@login_required
@role_required('analyst')
@log_user_activity
@measure_performance
def get_market_data(symbol):
    """Get detailed market data for a specific cryptocurrency"""
    try:
        # Validate symbol
        symbol = sanitize_input(symbol)
        if not validate_crypto_symbol(symbol):
            return jsonify({'error': 'Invalid cryptocurrency symbol'}), 400

        # Get timeframe from query params
        timeframe = request.args.get('timeframe', '1d')
        if not validate_timeframe(timeframe):
            timeframe = '1d'

        # Map symbol format for exchange
        symbol_map = {
            'BTC-USD': 'BTC/USDT',
            'ETH-USD': 'ETH/USDT',
            'BNB-USD': 'BNB/USDT',
            'ADA-USD': 'ADA/USDT',
            'SOL-USD': 'SOL/USDT'
        }
        exchange_symbol = symbol_map.get(symbol, symbol.replace('-', '/'))

        # Get crypto service data
        crypto_service = CryptoService()
        try:
            # Get current market data
            market_data = crypto_service.get_market_data([exchange_symbol])
            current_data = market_data[0] if market_data else {}

            # Get historical data using CoinGecko API
            timeframe_days_map = {
                '1d': 1,
                '1w': 7,
                '1m': 30,
                '3m': 90,
                '6m': 180,
                '1y': 365,
            }
            days = timeframe_days_map.get(timeframe, 7)

            historical_data = crypto_service.get_historical_data(
                exchange_symbol, days)

            # Format response
            response_data = {
                'symbol': symbol,
                'current': current_data,
                'historical': {
                    'datetime': [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in historical_data.index] if not historical_data.empty else [],
                    'open': historical_data['open'].tolist() if not historical_data.empty else [],
                    'high': historical_data['high'].tolist() if not historical_data.empty else [],
                    'low': historical_data['low'].tolist() if not historical_data.empty else [],
                    'close': historical_data['close'].tolist() if not historical_data.empty else [],
                    'volume': historical_data['volume'].tolist() if not historical_data.empty else []
                },
                'timeframe': timeframe,
                'last_updated': datetime.utcnow().isoformat()
            }

            return jsonify({
                'success': True,
                'data': response_data
            })

        except Exception as e:
            current_app.logger.error(
                f"Market data error for {symbol}: {str(e)}")
            return jsonify({'error': 'Failed to get market data'}), 500

    except Exception as e:
        current_app.logger.error(f"Market data API error: {str(e)}")
        return jsonify({'error': 'Ошибка при получении данных'}), 500


@analyst_bp.route('/api/search')
@login_required
@role_required('analyst')
@log_user_activity
@measure_performance
def search_cryptocurrencies():
    """Search for cryptocurrencies"""
    try:
        query = request.args.get('q', '').strip().upper()
        if not query or len(query) < 2:
            return jsonify({'results': []})

        # Search in available cryptocurrencies
        available_cryptos = [
            {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'rank': 1},
            {'symbol': 'ETH-USD', 'name': 'Ethereum', 'rank': 2},
            {'symbol': 'BNB-USD', 'name': 'Binance Coin', 'rank': 3},
            {'symbol': 'ADA-USD', 'name': 'Cardano', 'rank': 8},
            {'symbol': 'SOL-USD', 'name': 'Solana', 'rank': 5},
            {'symbol': 'DOT-USD', 'name': 'Polkadot', 'rank': 12},
            {'symbol': 'AVAX-USD', 'name': 'Avalanche', 'rank': 15},
            {'symbol': 'MATIC-USD', 'name': 'Polygon', 'rank': 18},
            {'symbol': 'LINK-USD', 'name': 'Chainlink', 'rank': 20},
            {'symbol': 'UNI-USD', 'name': 'Uniswap', 'rank': 25}
        ]

        # Filter based on query
        results = []
        for crypto in available_cryptos:
            if (query in crypto['symbol'] or
                query in crypto['name'].upper() or
                    crypto['name'].upper().startswith(query)):
                results.append(crypto)

        # Sort by rank
        results.sort(key=lambda x: x['rank'])

        return jsonify({
            'success': True,
            'results': results[:10]  # Limit to 10 results
        })

    except Exception as e:
        current_app.logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'Ошибка поиска'}), 500


@analyst_bp.route('/api/analysis_history')
@login_required
@role_required('analyst')
@log_user_activity
@measure_performance
def get_analysis_history():
    """Get user's analysis history"""
    try:
        # Get analysis metrics for the user
        analyses = SystemMetrics.query.filter_by(
            metric_name='analysis_request'
        ).filter(
            SystemMetrics.tags.contains({'user_id': current_user.id})
        ).order_by(SystemMetrics.created_at.desc()).limit(50).all()

        history = []
        for analysis in analyses:
            tags = analysis.tags or {}
            history.append({
                'id': analysis.id,
                'symbol': tags.get('symbol', 'Unknown'),
                'analysis_type': tags.get('analysis_type', 'Unknown'),
                'timeframe': tags.get('timeframe', 'Unknown'),
                'created_at': analysis.created_at.isoformat(),
                'success': True  # Assuming successful if recorded
            })

        return jsonify({
            'success': True,
            'history': history
        })

    except Exception as e:
        current_app.logger.error(f"Analysis history error: {str(e)}")
        return jsonify({'error': 'Ошибка при получении истории'}), 500
