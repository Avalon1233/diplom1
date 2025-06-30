import os
import requests # не трогать
import logging
import asyncio
import tempfile
import matplotlib
from dotenv import load_dotenv
load_dotenv()
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
)
import ccxt
import pandas as pd
from threading import Thread
import time
from config import Config
bot_token = Config.TELEGRAM_BOT_TOKEN


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SYMBOLS = [
    ("BTC/USDT", "Биткоин (BTC)"),
    ("ETH/USDT", "Эфириум (ETH)"),
    ("BNB/USDT", "Binance Coin (BNB)")
]
PERIODS = [
    ("1d", "1 день"),
    ("1w", "1 неделя"),
    ("1m", "1 месяц"),
    ("3m", "3 месяца"),
    ("1y", "1 год"),
]
MA_CHOICES = [
    ("7,30", "MA7 / MA30 (по умолчанию)"),
    ("10,50", "MA10 / MA50"),
    ("20,60", "MA20 / MA60")
]

# Для простоты используем файл (или dict) для подписчиков.
SUBSCRIBERS_FILE = "tg_subscribers.txt"

def load_subscribers():
    subs = {}
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 3:
                    chat_id, symbol, ma = parts
                    subs[int(chat_id)] = {"symbol": symbol, "ma": ma}
    return subs

def save_subscribers(subs):
    with open(SUBSCRIBERS_FILE, 'w', encoding='utf-8') as f:
        for chat_id, data in subs.items():
            f.write(f"{chat_id}|{data['symbol']}|{data['ma']}\n")

def subscribe(chat_id, symbol, ma):
    subs = load_subscribers()
    subs[chat_id] = {"symbol": symbol, "ma": ma}
    save_subscribers(subs)

def unsubscribe(chat_id):
    subs = load_subscribers()
    if chat_id in subs:
        del subs[chat_id]
        save_subscribers(subs)

def is_subscribed(chat_id):
    subs = load_subscribers()
    return chat_id in subs

def get_trend(symbol, period='1d', ma_choice="7,30"):
    try:
        ma_short, ma_long = map(int, ma_choice.split(','))
        exchange = ccxt.binance()
        timeframe_map = {'1d': '15m', '1w': '1h', '1m': '4h', '3m': '1d', '1y': '1d'}
        limit_map = {'1d': 96, '1w': 168, '1m': 180, '3m': 90, '1y': 365}
        timeframe = timeframe_map.get(period, '1h')
        limit = limit_map.get(period, 168)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df[f"MA{ma_short}"] = df['close'].rolling(ma_short).mean()
        df[f"MA{ma_long}"] = df['close'].rolling(ma_long).mean()
        ma_s = df[f"MA{ma_short}"].iloc[-1]
        ma_l = df[f"MA{ma_long}"].iloc[-1]
        last_close = df['close'].iloc[-1]

        if pd.isna(ma_s) or pd.isna(ma_l):
            trend = "Недостаточно данных"
            recommendation = "Нет рекомендации"
        elif ma_s > ma_l:
            trend = "Восходящий"
            recommendation = "Покупать или держать"
        elif ma_s < ma_l:
            trend = "Нисходящий"
            recommendation = "Продавать или не покупать"
        else:
            trend = "Боковой"
            recommendation = "Держать"
        return trend, recommendation, round(last_close, 2), df, ma_short, ma_long
    except Exception as e:
        logger.error(f"Ошибка при получении данных: {e}")
        return "Ошибка", "Не удалось получить данные", 0, pd.DataFrame(), 0, 0

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_menu(update, context)

async def show_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔎 Анализ тренда", callback_data='trend')],
        [InlineKeyboardButton("🔔 Подписка на уведомления", callback_data='subscribe_menu')],
        [InlineKeyboardButton("📈 Актуальные цены", callback_data='prices')],
        [InlineKeyboardButton("ℹ️ Инструкция", callback_data='help')],
        [InlineKeyboardButton("👨‍💻 О боте", callback_data='about')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.message:
        await update.message.reply_text("Главное меню:", reply_markup=reply_markup)
    else:
        await update.callback_query.edit_message_text("Главное меню:", reply_markup=reply_markup)

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    chat_id = query.message.chat_id

    if data == 'trend':
        keyboard = [
            [InlineKeyboardButton(name, callback_data=f"symbol_{symbol}")]
            for symbol, name in SYMBOLS
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='menu')])
        await query.edit_message_text("Выберите криптовалюту:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("symbol_"):
        symbol = data.split("_", 1)[1]
        context.user_data['symbol'] = symbol
        keyboard = [
            [InlineKeyboardButton(name, callback_data=f"period_{period}")]
            for period, name in PERIODS
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='trend')])
        await query.edit_message_text("Выберите период анализа:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("period_"):
        symbol = context.user_data.get('symbol', SYMBOLS[0][0])
        period = data.split("_", 1)[1]
        context.user_data['period'] = period
        keyboard = [
            [InlineKeyboardButton(name, callback_data=f"ma_{ma}")]
            for ma, name in MA_CHOICES
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"symbol_{symbol}")])
        await query.edit_message_text("Выберите параметры скользящих MA:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("ma_"):
        symbol = context.user_data.get('symbol', SYMBOLS[0][0])
        period = context.user_data.get('period', '1d')
        ma_choice = data.split("_", 1)[1]
        trend, recommendation, price, df, ma_s, ma_l = await asyncio.to_thread(get_trend, symbol, period, ma_choice)
        period_text = dict(PERIODS).get(period, period)
        name = dict(SYMBOLS).get(symbol, symbol)
        ma_label = f"MA{ma_s}/MA{ma_l}"

        message = (
            f'📊 <b>{name}</b> за период <b>{period_text}</b> ({ma_label}):\n'
            f'<b>Цена</b>: ${price}\n'
            f'<b>Тренд</b>: {trend}\n'
            f'<b>Рекомендация</b>: {recommendation}'
        )

        if df.empty or len(df) < max(ma_s, ma_l):
            await query.edit_message_text("Недостаточно данных для построения графика.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='trend')]]))
            return

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.figure(figsize=(9, 5))
            plt.plot(df['datetime'], df['close'], label='Цена')
            plt.plot(df['datetime'], df[f"MA{ma_s}"], label=f'MA{ma_s}')
            plt.plot(df['datetime'], df[f"MA{ma_l}"], label=f'MA{ma_l}')
            plt.title(f"{name} — {period_text} ({ma_label})")
            plt.xlabel('Дата')
            plt.ylabel('Цена, $')
            plt.legend()
            plt.tight_layout()
            plt.grid(alpha=0.2)
            plt.savefig(tmpfile.name)
            plt.close()
            tmpfile_path = tmpfile.name

        with open(tmpfile_path, 'rb') as img:
            await query.message.reply_photo(photo=InputFile(img), caption=message, parse_mode="HTML")

        try:
            os.remove(tmpfile_path)
        except Exception as e:
            logger.warning(f"Не удалось удалить временный файл: {e}")

        await show_menu(update, context)

    elif data == 'subscribe_menu':
        if is_subscribed(chat_id):
            await query.edit_message_text(
                "Вы уже подписаны на уведомления.\n"
                "Можем отправлять тренд раз в 6 часов по выбранной паре.\n"
                "Хотите отписаться?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("❌ Отписаться", callback_data="unsubscribe")],
                    [InlineKeyboardButton("⬅️ Назад", callback_data='menu')]
                ])
            )
        else:
            keyboard = [
                [InlineKeyboardButton(name, callback_data=f"subscribe_{symbol}")]
                for symbol, name in SYMBOLS
            ]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='menu')])
            await query.edit_message_text("Выберите криптовалюту для подписки:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("subscribe_"):
        symbol = data.split("_", 1)[1]
        context.user_data['sub_symbol'] = symbol
        keyboard = [
            [InlineKeyboardButton(name, callback_data=f"subscribe_ma_{ma}")]
            for ma, name in MA_CHOICES
        ]
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='subscribe_menu')])
        await query.edit_message_text("Выберите параметры MA для подписки:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("subscribe_ma_"):
        symbol = context.user_data.get('sub_symbol', SYMBOLS[0][0])
        ma_choice = data.split("_", 2)[2]
        subscribe(chat_id, symbol, ma_choice)
        await query.edit_message_text(
            f"✅ Подписка оформлена!\nБудете получать уведомления по {dict(SYMBOLS).get(symbol,symbol)} ({ma_choice})",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
        )

    elif data == "unsubscribe":
        unsubscribe(chat_id)
        await query.edit_message_text(
            "❌ Подписка отменена.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
        )

    elif data == 'prices':
        exchange = ccxt.binance()
        prices = []
        for symbol, name in SYMBOLS:
            try:
                ticker = exchange.fetch_ticker(symbol)
                prices.append(f"{name}: <b>${ticker['last']}</b>")
            except Exception:
                prices.append(f"{name}: <i>Ошибка получения цены</i>")
        await query.edit_message_text(
            "📈 <b>Актуальные цены:</b>\n" + "\n".join(prices),
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
        )

    elif data == 'help':
        await query.edit_message_text(
            "ℹ️ <b>Инструкция:</b>\n"
            "1. 'Анализ тренда' — пошагово выберите монету, период и параметры MA.\n"
            "2. 'Подписка на уведомления' — выберите монету и MA для автоматических алертов.\n"
            "3. После анализа получите график, тренд и рекомендацию.\n"
            "4. Для возврата — кнопка 'Назад'.",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
        )
    elif data == 'about':
        await query.edit_message_text(
            "🤖 <b>CryptoBot</b> — телеграм-бот для анализа трендов популярных криптовалют.\n"
            "Автоматические уведомления, графики, тренды, рекомендации — всё на русском!\n\n"
            "<i>Автор: 2025</i>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
        )
    elif data == 'menu':
        await show_menu(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Используйте /start или /menu для вызова главного меню.\n'
        'Для справки напишите /help.'
    )

def send_trend_notification(symbol, trend, recommendation, price, chat_id, period='1d', ma_choice="7,30"):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в окружении")
    name = dict(SYMBOLS).get(symbol, symbol)
    ma_label = f"MA{ma_choice.replace(',', '/MA')}"
    text = (
        f"📊 Тренд по {name} ({ma_label}):\n"
        f"Цена: ${price}\n"
        f"Тренд: {trend}\n"
        f"Рекомендация: {recommendation}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    response = requests.post(url, json=payload)
    if not response.ok:
        raise RuntimeError(f"Telegram API error: {response.text}")

def notification_daemon():
    while True:
        subs = load_subscribers()
        for chat_id, data in subs.items():
            symbol = data["symbol"]
            ma_choice = data["ma"]
            trend, recommendation, price, _, _, _ = get_trend(symbol, '1d', ma_choice)
            try:
                send_trend_notification(symbol, trend, recommendation, price, chat_id, '1d', ma_choice)
            except Exception as e:
                logger.error(f"Не удалось отправить уведомление {chat_id}: {e}")
        time.sleep(6*60*60)  # каждые 6 часов

def main():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("Не задан TELEGRAM_BOT_TOKEN в окружении.")
        return

    # Запуск демона уведомлений
    Thread(target=notification_daemon, daemon=True).start()

    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('menu', show_menu))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CallbackQueryHandler(menu_callback))

    logger.info("Бот запущен.")
    app.run_polling()

if __name__ == '__main__':
    main()
