# app/blueprints/trader.py
"""
Trader blueprint for trading dashboard, market data, and portfolio management
"""
from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for
from flask_login import login_required, current_user
from datetime import datetime

from app.models import TradingSession, PriceAlert, db, Cryptocurrency
from app.utils.decorators import role_required, cache_response, log_user_activity, measure_performance
from app.utils.validators import validate_crypto_symbol, validate_price
from app.services.crypto_service import CryptoService
from app.utils.security import sanitize_input
from sqlalchemy import func

trader_bp = Blueprint('trader', __name__, url_prefix='/trader')


@trader_bp.route('/dashboard')
@login_required
@role_required('trader')
@log_user_activity
@measure_performance
def dashboard():
    """Trader dashboard with portfolio overview and quick stats"""
    try:
        # Get user's trading sessions
        recent_sessions = TradingSession.query.filter_by(user_id=current_user.id)\
            .order_by(TradingSession.created_at.desc()).limit(5).all()

        # Get user's price alerts
        active_alerts = PriceAlert.query.filter_by(
            user_id=current_user.id,
            is_active=True
        ).order_by(PriceAlert.created_at.desc()).limit(10).all()

        # Get portfolio summary (demo data for now)
        portfolio_summary = {
            'total_value': 10000.00,  # Demo value
            'daily_change': 2.5,
            'total_change': 15.3,
            'positions': 5
        }

        # Get market overview
        crypto_service = CryptoService()
        try:
            market_data = crypto_service.get_market_data(
                ['BTC-USD', 'ETH-USD', 'BNB-USD'])
        except Exception as e:
            current_app.logger.error(f"Failed to get market data: {str(e)}")
            market_data = []

        return render_template('trader/dashboard.html',
                               recent_sessions=recent_sessions,
                               active_alerts=active_alerts,
                               portfolio_summary=portfolio_summary,
                               market_data=market_data)

    except Exception as e:
        current_app.logger.error(f"Trader dashboard error: {str(e)}")
        return render_template('trader/dashboard.html',
                               recent_sessions=[],
                               active_alerts=[],
                               portfolio_summary={},
                               market_data=[])


@trader_bp.route('/market')
@login_required
@role_required('trader')
@log_user_activity
@measure_performance
def market():
    """Redirects to the new market overview page."""
    return redirect(url_for('market.market_overview'))


@trader_bp.route('/api/alerts', methods=['GET', 'POST'])
@login_required
@role_required('trader')
@log_user_activity
@measure_performance
def manage_alerts():
    """Manage price alerts"""
    if request.method == 'GET':
        try:
            alerts = PriceAlert.query.filter_by(user_id=current_user.id)\
                .order_by(PriceAlert.created_at.desc()).all()
            return jsonify({'success': True, 'alerts': [alert.to_dict() for alert in alerts]})
        except Exception as e:
            current_app.logger.error(f"Get alerts error: {str(e)}")
            return jsonify({'error': 'Ошибка при получении алертов'}), 500

    elif request.method == 'POST':
        data = request.get_json()
        symbol = sanitize_input(data.get('symbol'))
        target_price = data.get('target_price')
        condition = sanitize_input(data.get('condition'))

        if not all([symbol, target_price, condition]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        is_valid, price_value, price_error = validate_price(str(target_price))
        if not is_valid:
            return jsonify({'success': False, 'error': price_error}), 400

        crypto = Cryptocurrency.query.filter(func.lower(Cryptocurrency.symbol) == symbol.lower()).first()
        if not crypto:
            return jsonify({'success': False, 'error': f'Cryptocurrency {symbol} not found'}), 404

        alert = PriceAlert(
            user_id=current_user.id,
            cryptocurrency_id=crypto.id,
            target_price=price_value,
            condition=condition
        )
        db.session.add(alert)
        db.session.commit()
        return jsonify({'success': True, 'alert': alert.to_dict()}), 201


@trader_bp.route('/api/alerts/<int:alert_id>', methods=['DELETE'])
@login_required
@role_required('trader')
@log_user_activity
@measure_performance
def delete_alert(alert_id):
    """Delete a price alert"""
    try:
        alert = PriceAlert.query.filter_by(id=alert_id, user_id=current_user.id).first_or_404()
        db.session.delete(alert)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Alert deleted successfully'})
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete alert error: {str(e)}")
        return jsonify({'error': 'Failed to delete alert'}), 500


@trader_bp.route('/api/market_search')
@login_required
@role_required('trader')
@cache_response(timeout=30)
@measure_performance
def market_search():
    """Search for cryptocurrencies in market data"""
    try:
        query = request.args.get('q', '').strip().upper()
        if not query or len(query) < 2:
            return jsonify({'results': []})

        # Search in predefined symbols
        all_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT',
            'ATOM/USDT', 'ALGO/USDT', 'XRP/USDT', 'LTC/USDT', 'BCH/USDT',
            'FIL/USDT', 'VET/USDT', 'TRX/USDT', 'EOS/USDT', 'XLM/USDT'
        ]

        matching_symbols = [s for s in all_symbols if query in s]

        if matching_symbols:
            crypto_service = CryptoService()
            try:
                results = crypto_service.get_market_data(matching_symbols[:10])
                return jsonify({'results': results})
            except Exception as e:
                current_app.logger.error(f"Market search error: {str(e)}")
                return jsonify({'results': []})

        return jsonify({'results': []})

    except Exception as e:
        current_app.logger.error(f"Market search error: {str(e)}")
        return jsonify({'error': 'Ошибка поиска'}), 500


@trader_bp.route('/api/portfolio')
@login_required
@role_required('trader')
@cache_response(timeout=60)
@measure_performance
def get_portfolio():
    """Get user's portfolio data with holdings and performance"""
    try:
        # Demo portfolio data - in real implementation, this would come from database
        portfolio_data = {
            'total_balance': 12345.67,
            'total_pnl': 1234.56,
            'total_pnl_percent': 12.5,
            'active_positions': 8,
            'open_orders': 3,
            'daily_change': 2.34,
            'daily_change_percent': 2.34,
            'holdings': [
                {
                    'symbol': 'BTC',
                    'name': 'Bitcoin',
                    'quantity': 0.15432,
                    'avg_price': 42000.00,
                    'current_price': 43250.00,
                    'value': 6670.00,
                    'pnl': 567.89,
                    'pnl_percent': 9.3,
                    'allocation': 54.1
                },
                {
                    'symbol': 'ETH',
                    'name': 'Ethereum',
                    'quantity': 1.8765,
                    'avg_price': 2500.00,
                    'current_price': 2650.00,
                    'value': 4972.25,
                    'pnl': 234.56,
                    'pnl_percent': 4.95,
                    'allocation': 40.3
                },
                {
                    'symbol': 'BNB',
                    'name': 'Binance Coin',
                    'quantity': 2.156,
                    'avg_price': 320.00,
                    'current_price': 310.50,
                    'value': 669.42,
                    'pnl': -23.45,
                    'pnl_percent': -3.38,
                    'allocation': 5.4
                }
            ],
            'recent_transactions': [
                {
                    'id': 1,
                    'type': 'buy',
                    'symbol': 'BTC/USDT',
                    'quantity': 0.025,
                    'price': 43250.00,
                    'total': 1081.25,
                    'timestamp': '2025-09-24T10:30:00Z',
                    'status': 'completed'
                },
                {
                    'id': 2,
                    'type': 'sell',
                    'symbol': 'ETH/USDT',
                    'quantity': 0.5,
                    'price': 2650.00,
                    'total': 1325.00,
                    'timestamp': '2025-09-24T09:15:00Z',
                    'status': 'pending'
                },
                {
                    'id': 3,
                    'type': 'buy',
                    'symbol': 'ADA/USDT',
                    'quantity': 1000,
                    'price': 0.50,
                    'total': 500.00,
                    'timestamp': '2025-09-24T08:45:00Z',
                    'status': 'completed'
                }
            ],
            'portfolio_history': [
                {'date': '2025-09-17', 'value': 10000.00},
                {'date': '2025-09-18', 'value': 10250.00},
                {'date': '2025-09-19', 'value': 10100.00},
                {'date': '2025-09-20', 'value': 10500.00},
                {'date': '2025-09-21', 'value': 11000.00},
                {'date': '2025-09-22', 'value': 11200.00},
                {'date': '2025-09-23', 'value': 12000.00},
                {'date': '2025-09-24', 'value': 12345.67}
            ]
        }

        return jsonify(portfolio_data)

    except Exception as e:
        current_app.logger.error(f"Portfolio API error: {str(e)}")
        return jsonify({'error': 'Ошибка получения данных портфолио'}), 500


@trader_bp.route('/api/watchlist')
@login_required
@role_required('trader')
@cache_response(timeout=30)
@measure_performance
def get_watchlist():
    """Get user's watchlist with real-time prices"""
    try:
        # Demo watchlist data - in real implementation, this would be user-specific
        crypto_service = CryptoService()

        watchlist_symbols = ['BTC/USDT', 'ETH/USDT',
                             'BNB/USDT', 'ADA/USDT', 'SOL/USDT']

        try:
            market_data = crypto_service.get_market_data(watchlist_symbols)

            watchlist_data = []
            for crypto in market_data:
                watchlist_data.append({
                    'symbol': crypto.get('symbol', 'N/A'),
                    'name': crypto.get('name', 'Unknown'),
                    'price': crypto.get('last', 0),
                    'change_24h': crypto.get('change', 0),
                    'change_percent_24h': crypto.get('percentage', 0),
                    'volume_24h': crypto.get('volume', 0),
                    'market_cap': crypto.get('market_cap', 0),
                    'rank': crypto.get('market_cap_rank', 0)
                })

            return jsonify({
                'watchlist': watchlist_data,
                'last_updated': datetime.utcnow().isoformat()
            })

        except Exception as e:
            current_app.logger.error(
                f"Failed to get watchlist market data: {str(e)}")
            # Return demo data if API fails
            demo_watchlist = [
                {
                    'symbol': 'BTC/USDT',
                    'name': 'Bitcoin',
                    'price': 43250.00,
                    'change_24h': 2.34,
                    'change_percent_24h': 2.34,
                    'volume_24h': 28500000000,
                    'market_cap': 850000000000,
                    'rank': 1
                },
                {
                    'symbol': 'ETH/USDT',
                    'name': 'Ethereum',
                    'price': 2650.00,
                    'change_24h': -1.23,
                    'change_percent_24h': -1.23,
                    'volume_24h': 15200000000,
                    'market_cap': 320000000000,
                    'rank': 2
                },
                {
                    'symbol': 'BNB/USDT',
                    'name': 'Binance Coin',
                    'price': 310.50,
                    'change_24h': 0.87,
                    'change_percent_24h': 0.87,
                    'volume_24h': 1200000000,
                    'market_cap': 47000000000,
                    'rank': 4
                }
            ]

            return jsonify({
                'watchlist': demo_watchlist,
                'last_updated': datetime.utcnow().isoformat()
            })

    except Exception as e:
        current_app.logger.error(f"Watchlist API error: {str(e)}")
        return jsonify({'error': 'Ошибка получения списка наблюдения'}), 500


@trader_bp.route('/api/chart/<symbol>')
@login_required
@role_required('trader')
@cache_response(timeout=300)
@measure_performance
def get_chart_data(symbol):
    """Get chart data for a specific cryptocurrency"""
    try:
        timeframe = request.args.get('timeframe', '1d')
        limit = request.args.get('limit', '100')

        # Demo chart data - in real implementation, this would come from CoinGecko or similar
        import random
        from datetime import datetime, timedelta

        # Generate demo OHLCV data
        chart_data = []
        base_price = 43000 if 'BTC' in symbol.upper(
        ) else 2600 if 'ETH' in symbol.upper() else 300
        current_time = datetime.utcnow()

        for i in range(int(limit)):
            timestamp = current_time - timedelta(hours=i)

            # Generate realistic OHLCV data
            price_variation = random.uniform(-0.05, 0.05)  # ±5% variation
            open_price = base_price * (1 + price_variation)

            high_variation = random.uniform(0, 0.03)  # 0-3% higher than open
            low_variation = random.uniform(-0.03, 0)  # 0-3% lower than open
            close_variation = random.uniform(-0.02, 0.02)  # ±2% from open

            high_price = open_price * (1 + high_variation)
            low_price = open_price * (1 + low_variation)
            close_price = open_price * (1 + close_variation)
            volume = random.uniform(1000000, 10000000)

            chart_data.append({
                'timestamp': timestamp.isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2)
            })

            base_price = close_price  # Use close as next base price

        # Reverse to get chronological order
        chart_data.reverse()

        return jsonify({
            'symbol': symbol,
            'timeframe': timeframe,
            'data': chart_data,
            'last_updated': datetime.utcnow().isoformat()
        })

    except Exception as e:
        current_app.logger.error(f"Chart data API error: {str(e)}")
        return jsonify({'error': 'Ошибка получения данных графика'}), 500
