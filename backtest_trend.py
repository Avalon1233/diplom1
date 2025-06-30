import ccxt
import pandas as pd
import numpy as np
import random
import os
import matplotlib

# Используем 'Agg' для headless сохранения
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_binance_ohlcv(symbol, timeframe='1h', limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_trend_and_recommendation(df):
    ma7 = df['close'].rolling(window=7).mean()
    ma30 = df['close'].rolling(window=30).mean()
    last_ma7 = ma7.iloc[-1]
    last_ma30 = ma30.iloc[-1]
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2] if len(df) > 1 else last_close

    if pd.isna(last_ma7) or pd.isna(last_ma30):
        trend = "Недостаточно данных"
        recommendation = "Нет рекомендации"
    elif (last_ma7 > last_ma30) and (last_close > prev_close):
        trend = "Восходящий"
        recommendation = "Покупать или держать"
    elif (last_ma7 < last_ma30) and (last_close < prev_close):
        trend = "Нисходящий"
        recommendation = "Продавать или не покупать"
    elif abs(last_ma7 - last_ma30) / (last_ma30 + 1e-9) < 0.003:
        trend = "Боковой"
        recommendation = "Держать"
    elif last_ma7 > last_ma30:
        trend = "Восходящий"
        recommendation = "Держать"
    elif last_ma7 < last_ma30:
        trend = "Нисходящий"
        recommendation = "Держать или продавать"
    else:
        trend = "Неопределён"
        recommendation = "Нет рекомендации"
    return trend, recommendation

def calculate_max_drawdown(balance_history):
    max_drawdown = 0
    peak = balance_history[0]
    for x in balance_history:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    return max_drawdown

def backtest(symbol='BTC/USDT', timeframe='1d', limit=365,
             stop_loss=0.07, take_profit=0.15, print_trades=False):
    df = get_binance_ohlcv(symbol, timeframe, limit)
    df['MA_7'] = df['close'].rolling(window=7).mean()
    df['MA_30'] = df['close'].rolling(window=30).mean()

    balance = 1000.0
    in_position = False
    position_size = 0.0
    entry_price = 0.0
    trades = []
    signals = []
    balance_history = [balance]

    for i in range(30, len(df)):
        trend, recommendation = get_trend_and_recommendation(df.iloc[:i+1])
        price = df['close'].iloc[i]
        dt = df['datetime'].iloc[i]

        # Стратегия: открываем только на весь баланс, 1 сделка за раз
        if in_position:
            change = (price - entry_price) / entry_price
            unrealized = position_size * (price - entry_price)
            # Стоп-лосс или тейк-профит
            if change <= -stop_loss:
                balance += position_size * (price - entry_price)
                trades.append(position_size * (price - entry_price))
                signals.append(('Stop-Loss', dt, price))
                in_position = False
                position_size = 0
                if print_trades:
                    print(f"{dt} STOP-LOSS: Продажа по {price:.2f}, П/У: {unrealized:.2f}")
            elif change >= take_profit:
                balance += position_size * (price - entry_price)
                trades.append(position_size * (price - entry_price))
                signals.append(('Take-Profit', dt, price))
                in_position = False
                position_size = 0
                if print_trades:
                    print(f"{dt} TAKE-PROFIT: Продажа по {price:.2f}, П/У: {unrealized:.2f}")

        if not in_position and recommendation.startswith('Покупать'):
            # Покупка на весь баланс (без плеча!)
            position_size = balance / price
            entry_price = price
            in_position = True
            signals.append(('Buy', dt, price))
            if print_trades:
                print(f"{dt} BUY {position_size:.5f} по {price:.2f}")
        elif in_position and 'Продавать' in recommendation:
            profit = position_size * (price - entry_price)
            balance += profit
            trades.append(profit)
            signals.append(('Sell', dt, price))
            in_position = False
            position_size = 0
            if print_trades:
                print(f"{dt} SELL {profit:.2f} по {price:.2f}")
        else:
            signals.append(('Hold', dt, price))

        # Фиксируем эквити (текущий баланс + нереализованный)
        if in_position:
            balance_history.append(balance + position_size * (price - entry_price))
        else:
            balance_history.append(balance)
        # Стоп-стратегии, если баланс ушёл в минус
        if balance <= 0:
            print("Депозит обнулён! Остановка торговли.")
            break

    # Если в позиции в конце теста — закрываем
    if in_position:
        price = df['close'].iloc[-1]
        profit = position_size * (price - entry_price)
        balance += profit
        trades.append(profit)
        signals.append(('Close', df['datetime'].iloc[-1], price))
        balance_history.append(balance)

    print(f"\n=== STRATEGY: MA7/MA30 (без плеча, весь баланс) ===")
    print(f"Всего сделок: {len(trades)}")
    print(f"Суммарная прибыль: {sum(trades):.2f} $")
    print(f"Итоговый баланс: {balance:.2f} $")
    print(f"Средняя прибыль на сделку: {np.mean(trades) if trades else 0:.2f} $")
    print(f"Максимальная просадка: {calculate_max_drawdown(balance_history)*100:.2f}%")

    # Buy&Hold
    start_price = df['close'].iloc[30]
    end_price = df['close'].iloc[-1]
    buyhold_balance = 1000 * (end_price / start_price)
    print(f"\n=== СТРАТЕГИЯ КУПИ-ДЕРЖИ ===")
    print(f"Итоговый баланс buy&hold: {buyhold_balance:.2f} $")
    print(f"Профит buy&hold: {buyhold_balance-1000:.2f} $")

    # Случайная стратегия (тоже на весь баланс)
    random_balance = 1000
    random_in_pos = False
    random_entry = 0
    random_pos_size = 0
    random_trades = []
    for i in range(30, len(df)):
        price = df['close'].iloc[i]
        if not random_in_pos and random.random() < 0.05:
            random_pos_size = random_balance / price
            random_entry = price
            random_in_pos = True
        elif random_in_pos and random.random() < 0.05:
            profit = random_pos_size * (price - random_entry)
            random_balance += profit
            random_trades.append(profit)
            random_in_pos = False
    if random_in_pos:
        profit = random_pos_size * (df['close'].iloc[-1] - random_entry)
        random_balance += profit
        random_trades.append(profit)
    print(f"\n=== СЛУЧАЙНАЯ СТРАТЕГИЯ ===")
    print(f"Итоговый баланс: {random_balance:.2f} $")
    print(f"Профит: {random_balance-1000:.2f} $")

    # Сохраняем график баланса
    # График баланса
    x = df['datetime'].iloc[30:30 + len(balance_history)]  # x: даты
    y = balance_history[:len(x)]  # y: баланс (точно такая же длина)
    plt.figure(figsize=(15, 6))
    plt.plot(x, y, label='MA Strategy')
    start_price = df['close'].iloc[30]
    plt.plot(x, [1000 * (p / start_price) for p in df['close'].iloc[30:30 + len(x)]], label='Buy&Hold')
    plt.title(f"Backtest {symbol} {timeframe}, стартовый баланс: 1000$")
    plt.xlabel("Дата")
    plt.ylabel("Баланс, $")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("balance.png")
    plt.close()
    # Сохраняем график входов/выходов
    plt.figure(figsize=(15,6))
    plt.plot(df['datetime'], df['close'], label='Цена')
    buys = [dt for signal, dt, price in signals if signal == 'Buy']
    buy_prices = [price for signal, dt, price in signals if signal == 'Buy']
    sells = [dt for signal, dt, price in signals if 'Sell' in signal or 'Take-Profit' in signal or 'Stop-Loss' in signal or signal == 'Close']
    sell_prices = [price for signal, dt, price in signals if 'Sell' in signal or 'Take-Profit' in signal or 'Stop-Loss' in signal or signal == 'Close']
    plt.scatter(buys, buy_prices, marker='^', color='g', label='Покупка')
    plt.scatter(sells, sell_prices, marker='v', color='r', label='Продажа/Закрытие')
    plt.title(f"Backtest {symbol} {timeframe}, точки входа/выхода")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("trades.png")
    plt.close()

    # Открыть графики (только для Windows)
    try:
        os.startfile("balance.png")
        os.startfile("trades.png")
    except Exception as e:
        print(f"Не удалось автоматически открыть файлы: {e}")

if __name__ == '__main__':
    backtest(symbol='BTC/USDT', timeframe='1d', limit=365)
