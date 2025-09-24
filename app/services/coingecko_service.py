from pycoingecko import CoinGeckoAPI
import time
import os
from flask import current_app
from app import cache


class CoinGeckoService:
    def __init__(self):
        self.cg = CoinGeckoAPI()
        self.request_delay = float(os.getenv('COINGECKO_REQUEST_DELAY', '1.2'))  # Respect API rate limits
        self.last_request_time = 0

    def _rate_limit(self):
        """Ensures we don't hit the CoinGecko rate limit."""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()

    @cache.memoize(timeout=3600) # Cache for 1 hour
    def get_coin_list(self):
        """Get list of all available coins."""
        self._rate_limit()
        try:
            return self.cg.get_coins_list()
        except Exception as e:
            current_app.logger.error(f'CoinGecko API error in get_coin_list: {e}')
            raise

    @cache.memoize(timeout=60) # Cache for 1 minute
    def get_coin_market_data(self, vs_currency='usd', per_page=100, page=1, ids=None):
        """Get market data for coins."""
        self._rate_limit()
        try:
            return self.cg.get_coins_markets(
                vs_currency=vs_currency,
                per_page=per_page,
                page=page,
                price_change_percentage='24h,7d,30d',
                order='market_cap_desc',
                ids=ids
            )
        except Exception as e:
            current_app.logger.error(f'CoinGecko API error in get_coin_market_data: {e}')
            raise

    @cache.memoize(timeout=300) # Cache for 5 minutes
    def get_coin_historical_data(self, coin_id, vs_currency='usd', days=30):
        """Get historical data for a coin."""
        self._rate_limit()
        try:
            return self.cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
        except Exception as e:
            current_app.logger.error(f'CoinGecko API error in get_coin_historical_data: {e}')
            raise

    @cache.memoize(timeout=300) # Cache for 5 minutes
    def get_coin_info(self, coin_id):
        """Get detailed information about a coin."""
        self._rate_limit()
        try:
            return self.cg.get_coin_by_id(id=coin_id)
        except Exception as e:
            current_app.logger.error(f'CoinGecko API error in get_coin_info: {e}')
            raise

    @cache.memoize(timeout=60) # Cache for 1 minute
    def get_coin_price(self, coin_id, vs_currency='usd'):
        """Get the current price of a coin."""
        self._rate_limit()
        try:
            return self.cg.get_price(ids=coin_id, vs_currencies=vs_currency)
        except Exception as e:
            current_app.logger.error(f'CoinGecko API error in get_coin_price: {e}')
            raise


# Create a global instance of the service
coingecko_service = CoinGeckoService()
