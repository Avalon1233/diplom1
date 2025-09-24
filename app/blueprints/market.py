# app/blueprints/market.py
from flask import Blueprint, render_template, jsonify, request, current_app
from flask_login import login_required, current_user

from app import db
from app.models import Cryptocurrency
from app.services.coingecko_service import coingecko_service

market_bp = Blueprint('market', __name__)

@market_bp.route('/')
@login_required
def market_overview():
    """Отображает главную страницу нового маркета."""
    return render_template('market/market.html')

# --- API для Избранного (Watchlist) ---

@market_bp.route('/api/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    """Получить список избранных криптовалют пользователя."""
    watchlist_coins = current_user.watchlist
    coin_ids = [coin.coin_id for coin in watchlist_coins]

    if not coin_ids:
        return jsonify({'success': True, 'data': []})

    try:
        # Получаем актуальные рыночные данные для избранных монет
        market_data = coingecko_service.get_coin_market_data_by_ids(coin_ids)
        return jsonify({'success': True, 'data': market_data})
    except Exception as e:
        current_app.logger.error(f"Error fetching watchlist data: {e}")
        return jsonify({'success': False, 'error': 'Could not fetch watchlist data'}), 500


@market_bp.route('/api/watchlist/toggle', methods=['POST'])
@login_required
def toggle_watchlist():
    """Добавить или удалить криптовалюту из избранного."""
    data = request.get_json()
    coin_id = data.get('coin_id')

    if not coin_id:
        return jsonify({'success': False, 'error': 'Coin ID is required'}), 400

    # Находим или создаем запись о криптовалюте в нашей БД
    crypto = Cryptocurrency.query.filter_by(coin_id=coin_id).first()
    if not crypto:
        try:
            # Если монеты нет, получаем инфо и добавляем
            coin_info = coingecko_service.get_coin_info(coin_id)
            crypto = Cryptocurrency(
                coin_id=coin_id,
                symbol=coin_info['symbol'],
                name=coin_info['name']
            )
            db.session.add(crypto)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(f"Could not add new crypto {coin_id} to DB: {e}")
            db.session.rollback()
            return jsonify({'success': False, 'error': 'Could not retrieve coin information'}), 500

    # Проверяем, есть ли монета в избранном, и выполняем действие
    if crypto in current_user.watchlist:
        current_user.watchlist.remove(crypto)
        action = 'removed'
    else:
        current_user.watchlist.append(crypto)
        action = 'added'

    try:
        db.session.commit()
        return jsonify({'success': True, 'action': action, 'coin_id': coin_id})
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating watchlist for user {current_user.id}: {e}")
        return jsonify({'success': False, 'error': 'Database error'}), 500

# --- API для Рыночных Данных ---

@market_bp.route('/api/coins')
@login_required
def get_coins():
    """Получить список монет с пагинацией."""
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 100)

    try:
        data = coingecko_service.get_coin_market_data(
            vs_currency='usd',
            per_page=per_page,
            page=page
        )
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        current_app.logger.error(f'Error fetching coins: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500


@market_bp.route('/api/coin/<string:coin_id>')
@login_required
def get_coin_details(coin_id):
    """Получить всю детальную информацию о монете."""
    try:
        # Вся информация, включая тикеры, уже есть в get_coin_info
        coin_info = coingecko_service.get_coin_info(coin_id)
        historical_data = coingecko_service.get_coin_historical_data(coin_id, days=30)

        # Данные по тикерам находятся внутри coin_info
        tickers = coin_info.get('tickers', [])

        response_data = {
            'info': coin_info,
            'history': historical_data,
            'tickers': tickers  # Теперь это корректные данные
        }

        return jsonify({'success': True, 'data': response_data})
    except Exception as e:
        current_app.logger.error(f'Error fetching details for coin {coin_id}: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500
