# app/blueprints/api.py
"""
Comprehensive API blueprint with enhanced security and functionality
"""
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required

from app import db, limiter
from app.models import PriceAlert
from app.utils.decorators import (
    role_required, measure_performance, cache_response,
    validate_json, handle_exceptions, log_user_activity
)
from app.services.crypto_service import CryptoService
from app.services.analysis_service import AnalysisService
from app.utils.validators import validate_price, validate_crypto_symbol

api_bp = Blueprint('api', __name__)

# Initialize services
crypto_service = CryptoService()
analysis_service = AnalysisService()


@api_bp.route('/health')
@limiter.limit("100 per minute")
def health():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '2.1.0' # Version bump
    })


@api_bp.route('/market-data')
@login_required
@limiter.limit("30 per minute")
@measure_performance
@cache_response(timeout=60)
@handle_exceptions
def market_data():
    """Get rich market data for popular cryptocurrencies."""
    symbols = request.args.getlist('symbols') or [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD',
        'SOL-USD', 'AVAX-USD', 'DOT-USD', 'LINK-USD'
    ]
    market_data = crypto_service.get_market_data(symbols)

    return jsonify({
        'success': True,
        'data': market_data,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'count': len(market_data),
        'source': 'CoinGecko API'
    })


@api_bp.route('/crypto/<string:symbol>')
@login_required
@limiter.limit("60 per minute")
@measure_performance
@cache_response(timeout=120)
@handle_exceptions
def get_crypto_data(symbol):
    """Get detailed data for a specific cryptocurrency."""
    if not validate_crypto_symbol(symbol):
        return jsonify({'success': False, 'error': 'Invalid symbol format'}), 400

    # Use the main market_data function which is now rich enough
    crypto_data = crypto_service.get_market_data([symbol])

    if not crypto_data:
        return jsonify({'success': False, 'error': 'Symbol not found'}), 404

    return jsonify({
        'success': True,
        'data': crypto_data[0],
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source': 'CoinGecko API'
    })


@api_bp.route('/crypto-prices')
@login_required
@limiter.limit("30 per minute")
@measure_performance
@cache_response(timeout=60)
@handle_exceptions
def crypto_prices():
    """Get current prices for multiple cryptocurrencies."""
    symbols = request.args.getlist('symbols') or ['BTC-USD', 'ETH-USD', 'BNB-USD']
    market_data = crypto_service.get_market_data(symbols)
    
    prices = {item['symbol']: item['current_price'] for item in market_data if 'current_price' in item}

    return jsonify({
        'success': True,
        'data': prices,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source': 'CoinGecko API'
    })


@api_bp.route('/ohlcv')
@login_required
@limiter.limit("20 per minute")
@measure_performance
@cache_response(timeout=120)
@handle_exceptions
def ohlcv_data():
    """Get historical OHLCV data for charting from Binance."""
    symbol = request.args.get('symbol', 'BTC-USD')
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 100))

    if not validate_crypto_symbol(symbol):
        return jsonify({'success': False, 'error': 'Invalid symbol format'}), 400

    df = crypto_service.get_binance_ohlcv(symbol, timeframe, limit)

    if df.empty:
        return jsonify({'success': False, 'error': 'Failed to fetch OHLCV data'}), 500

    # Convert DataFrame to JSON serializable format
    data = {
        'timestamps': [ts.isoformat() for ts in df.index],
        'opens': df['open'].tolist(),
        'highs': df['high'].tolist(),
        'lows': df['low'].tolist(),
        'closes': df['close'].tolist(),
        'volumes': df['volume'].tolist(),
    }

    return jsonify({
        'success': True,
        'data': data,
        'symbol': symbol,
        'timeframe': timeframe,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'source': 'Binance API (via ccxt)'
    })


@api_bp.route('/analyze', methods=['POST'])
@login_required
@role_required('analyst', 'admin')
@limiter.limit("10 per minute")
@measure_performance
@validate_json('symbol', 'timeframe')
@log_user_activity(activity_type='crypto_analysis')
@handle_exceptions
def analyze():
    """Perform advanced cryptocurrency analysis."""
    data = request.get_json()
    symbol = data['symbol']
    timeframe = data['timeframe']

    if not validate_crypto_symbol(symbol):
        return jsonify({'success': False, 'error': 'Invalid symbol format'}), 400

    result = analysis_service.advanced_ml_analysis(symbol, timeframe)

    return jsonify(result)


# Price Alert endpoints remain largely the same, but we'll audit them for correctness.

@api_bp.route('/alerts', methods=['GET'])
@login_required
@limiter.limit("30 per minute")
@measure_performance
@handle_exceptions
def get_alerts():
    """Get user's price alerts"""
    alerts = PriceAlert.query.filter_by(
        user_id=current_user.id,
        is_active=True
    ).order_by(PriceAlert.created_at.desc()).all()

    return jsonify({
        'success': True,
        'data': [alert.to_dict() for alert in alerts],
        'count': len(alerts),
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@api_bp.route('/alerts', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
@measure_performance
@validate_json('symbol', 'condition', 'target_price')
@log_user_activity(activity_type='create_price_alert')
@handle_exceptions
def create_alert():
    """Create a new price alert"""
    data = request.get_json()
    symbol = data['symbol']
    condition = data['condition']
    target_price = data['target_price']

    if not validate_crypto_symbol(symbol):
        return jsonify({'success': False, 'error': 'Invalid symbol'}), 400

    if condition not in ['above', 'below']:
        return jsonify({'success': False, 'error': 'Invalid condition. Must be "above" or "below"'}), 400

    is_valid, price_value, price_error = validate_price(str(target_price))
    if not is_valid:
        return jsonify({'success': False, 'error': 'Invalid price', 'message': price_error}), 400

    # Logic to get cryptocurrency ID from symbol
    from app.models import Cryptocurrency
    crypto = Cryptocurrency.query.filter_by(symbol=symbol.split('-')[0].lower()).first()
    if not crypto:
        return jsonify({'success': False, 'error': f'Cryptocurrency {symbol} not found in database'}), 404

    existing_alert = PriceAlert.query.filter_by(
        user_id=current_user.id,
        cryptocurrency_id=crypto.id,
        condition=condition,
        target_price=price_value,
        is_active=True
    ).first()

    if existing_alert:
        return jsonify({'success': False, 'error': 'Alert already exists'}), 409

    alert = PriceAlert(
        user_id=current_user.id,
        cryptocurrency_id=crypto.id,
        condition=condition,
        target_price=price_value,
        notify_telegram=data.get('notify_telegram', True),
        notify_email=data.get('notify_email', False)
    )

    db.session.add(alert)
    db.session.commit()

    return jsonify({
        'success': True,
        'data': alert.to_dict(),
        'message': 'Alert created successfully'
    }), 201


@api_bp.route('/alerts/<int:alert_id>', methods=['DELETE'])
@login_required
@limiter.limit("10 per minute")
@measure_performance
@log_user_activity(activity_type='delete_price_alert')
@handle_exceptions
def delete_alert(alert_id):
    """Delete a price alert."""
    alert = PriceAlert.query.filter_by(id=alert_id, user_id=current_user.id).first()

    if not alert:
        return jsonify({'success': False, 'error': 'Alert not found or unauthorized'}), 404

    db.session.delete(alert)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Alert deleted successfully'})
