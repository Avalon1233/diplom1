# app/services/crypto_service.py
"""
Cryptocurrency data service using ccxt for high-quality OHLC data and CoinGecko for market overview.
"""
import pandas as pd
import pytz
import ccxt
from datetime import datetime, timezone
from typing import List, Dict
from flask import current_app

from app import cache
from app.interfaces import CryptoServiceInterface
from app.services.coingecko_service import CoinGeckoService


class CryptoService(CryptoServiceInterface):
    """Service for cryptocurrency data using ccxt for OHLC and CoinGecko for market data."""

    def __init__(self):
        self.coingecko = CoinGeckoService()
        self.binance = ccxt.binance()
        self.symbol_mapping = self._get_symbol_mapping()

    def _get_symbol_mapping(self):
        """Centralized mapping for various symbol formats."""
        return {
            'BTC-USD': {'coingecko_id': 'bitcoin', 'binance_symbol': 'BTC/USDT'},
            'ETH-USD': {'coingecko_id': 'ethereum', 'binance_symbol': 'ETH/USDT'},
            'BNB-USD': {'coingecko_id': 'binancecoin', 'binance_symbol': 'BNB/USDT'},
            'XRP-USD': {'coingecko_id': 'ripple', 'binance_symbol': 'XRP/USDT'},
            'ADA-USD': {'coingecko_id': 'cardano', 'binance_symbol': 'ADA/USDT'},
            'SOL-USD': {'coingecko_id': 'solana', 'binance_symbol': 'SOL/USDT'},
            'AVAX-USD': {'coingecko_id': 'avalanche-2', 'binance_symbol': 'AVAX/USDT'},
            'DOT-USD': {'coingecko_id': 'polkadot', 'binance_symbol': 'DOT/USDT'},
            'LINK-USD': {'coingecko_id': 'chainlink', 'binance_symbol': 'LINK/USDT'},
            'UNI-USD': {'coingecko_id': 'uniswap', 'binance_symbol': 'UNI/USDT'},
        }

    def _resolve_symbol(self, symbol: str, target: str = 'binance_symbol') -> str:
        """Resolves a symbol to the required format (e.g., for Binance or CoinGecko)."""
        if symbol in self.symbol_mapping:
            return self.symbol_mapping[symbol][target]
        # Fallback for symbols not in the primary map
        if target == 'binance_symbol':
            return symbol.replace('-', '/') + 'T'
        else:  # coingecko_id
            return symbol.lower().split('-')[0]

    @cache.memoize(timeout=300)
    def get_binance_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 365) -> pd.DataFrame:
        """Get historical OHLCV data from Binance as a DataFrame."""
        try:
            binance_symbol = self._resolve_symbol(symbol, 'binance_symbol')
            
            # CCXT uses milliseconds for since parameter
            # Fetch more data than needed to ensure we have enough after filtering
            since = self.binance.parse8601(pd.Timestamp.now(tz='UTC') - pd.DateOffset(days=limit * 2))

            ohlcv = self.binance.fetch_ohlcv(binance_symbol, timeframe, since=since, limit=limit + 100)

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            # Ensure we return the correct number of rows from the end
            return df.iloc[-limit:]

        except Exception as e:
            current_app.logger.error(f"Error fetching Binance OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    @cache.memoize(timeout=60)
    def get_market_data(self, symbols: List[str]) -> List[Dict]:
        """Get rich market overview data for multiple symbols using CoinGecko API."""
        coingecko_ids = [self._resolve_symbol(s, 'coingecko_id') for s in symbols]
        try:
            market_data = self.coingecko.get_coin_market_data(ids=coingecko_ids)
            data_map = {d['id']: d for d in market_data}

            result = []
            for symbol in symbols:
                cg_id = self._resolve_symbol(symbol, 'coingecko_id')
                data = data_map.get(cg_id)
                if data:
                    result.append({
                        'symbol': symbol,
                        'name': data.get('name'),
                        'image': data.get('image'),
                        'current_price': data.get('current_price', 0),
                        'market_cap': data.get('market_cap', 0),
                        'market_cap_rank': data.get('market_cap_rank', 0),
                        'total_volume': data.get('total_volume', 0),
                        'high_24h': data.get('high_24h', 0),
                        'low_24h': data.get('low_24h', 0),
                        'price_change_24h': data.get('price_change_24h', 0),
                        'price_change_percentage_24h': data.get('price_change_percentage_24h', 0),
                    })
            return result
        except Exception as e:
            current_app.logger.error(f"Error fetching market data from CoinGecko: {e}")
            return []

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols from the mapping."""
        return list(self.symbol_mapping.keys())

    @cache.memoize(timeout=300)
    def get_coin_historical_data_df(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Compatibility method expected by AnalysisService.
        Returns daily OHLCV data DataFrame for the last `days` days.
        Columns: ['open', 'high', 'low', 'close', 'volume'] indexed by UTC datetime.
        """
        try:
            df = self.get_binance_ohlcv(symbol, timeframe='1d', limit=days)
            if df.empty:
                return df
            # Ensure correct dtypes
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            return df
        except Exception as e:
            current_app.logger.error(f"Error building historical data DataFrame for {symbol}: {e}")
            return pd.DataFrame()

