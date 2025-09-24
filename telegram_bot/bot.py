import os
import requests
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
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
)
from telegram.error import BadRequest
import ccxt
import pandas as pd
from config import Config
from html import escape
from app import app, db, User, PriceAlert
from notify import send_trend_notification as notify_from_module
from threading import Thread

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Глобальный экземпляр Application ---
application = ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).build()
# -------------------------------------

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


# ---- Работа с базой только внутри app.app_context() ----
def is_subscribed(chat_id):
    user = User.query.filter_by(telegram_chat_id=str(chat_id)).first()
    return user.is_tg_subscribed if user else False

def subscribe(chat_id, symbol, ma):
    user = User.query.filter_by(telegram_chat_id=str(chat_id)).first()
    if not user:
        logger.warning(f"Попытка подписки: chat_id={chat_id}, но пользователя нет в БД")
        return
    user.is_tg_subscribed = True
    user.tg_symbol = symbol
    user.tg_ma = ma
    db.session.commit()
    logger.info(f"Пользователь {chat_id} подписан: {symbol}, {ma}")

def unsubscribe(chat_id):
    user = User.query.filter_by(telegram_chat_id=str(chat_id)).first()
    if not user:
        logger.warning(f"Попытка отписки: chat_id={chat_id}, но пользователя нет в БД")
        return
    user.is_tg_subscribed = False
    db.session.commit()
    logger.info(f"Пользователь {chat_id} отписан")

def get_all_subscribers():
    return User.query.filter_by(is_tg_subscribed=True).all()
# --------------------------------------------------------

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
        [InlineKeyboardButton("👤 Личный кабинет", callback_data='profile')],
        [InlineKeyboardButton("ℹ️ Инструкция", callback_data='help')],
        [InlineKeyboardButton("📌 Алерты по цене", callback_data='alerts')],
        [InlineKeyboardButton("👨‍💻 О боте", callback_data='about')]
    ]
    try:
        reply_markup = InlineKeyboardMarkup(keyboard)
        if update.message:
            await update.message.reply_text("Главное меню:", reply_markup=reply_markup)
        else:
            await update.callback_query.edit_message_text("Главное меню:", reply_markup=reply_markup)
    except BadRequest as e:
        logger.error(f"BadRequest in show_menu: {e}")
        if "Message is not modified" not in str(e):
            raise


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    chat_id = update.effective_chat.id

    try:
        # ---- Подписка ----
        if data == 'subscribe_menu':
            with app.app_context():
                if is_subscribed(chat_id):
                    await query.edit_message_text(
                        "Вы уже подписаны на уведомления.\nХотите отписаться?",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("❌ Отписаться", callback_data="unsubscribe")],
                            [InlineKeyboardButton("⬅️ Назад", callback_data='menu')]
                        ])
                    )
                else:
                    keyboard = [
                        [InlineKeyboardButton(escape(name), callback_data=f"subscribe_symbol_{symbol}")]
                        for symbol, name in SYMBOLS
                    ]
                    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='menu')])
                    await query.edit_message_text(
                        "Выберите криптовалюту для подписки:",
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )

        elif data.startswith("subscribe_symbol_"):
            symbol = data.split("_", 2)[2]
            context.user_data['sub_symbol'] = symbol
            keyboard = [
                [InlineKeyboardButton(escape(name), callback_data=f"subscribe_period_{period}")]
                for period, name in PERIODS
            ]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='subscribe_menu')])
            await query.edit_message_text(
                "Выберите период для подписки:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        elif data.startswith("subscribe_period_"):
            period = data.split("_", 2)[2]
            context.user_data['sub_period'] = period
            keyboard = [
                [InlineKeyboardButton(escape(name), callback_data=f"subscribe_ma_{ma}")]
                for ma, name in MA_CHOICES
            ]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data=f'subscribe_symbol_{context.user_data["sub_symbol"]}')])
            await query.edit_message_text(
                "Выберите параметры MA для подписки:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )

        elif data.startswith("subscribe_ma_"):
            symbol = context.user_data.get('sub_symbol', SYMBOLS[0][0])
            period = context.user_data.get('sub_period', '1d')
            ma_choice = data.split("_", 2)[2]

            with app.app_context():
                user = User.query.filter_by(telegram_chat_id=str(chat_id)).first()
                if not user:
                    await query.edit_message_text(
                        f"❗ Ваш Telegram не привязан к профилю.\n\n"
                        f"1. Зарегистрируйтесь на сайте.\n"
                        f"2. В личном кабинете укажите этот chat_id:\n"
                        f"<code>{escape(str(chat_id))}</code>\n"
                        f"3. После этого попробуйте снова оформить подписку через Telegram.",
                        parse_mode="HTML",
                        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
                    )
                    return

                user.is_tg_subscribed = True
                user.tg_symbol = symbol
                user.tg_ma = ma_choice
                user.tg_period = period
                db.session.commit()

            await query.edit_message_text(
                f"✅ Подписка оформлена!\n"
                f"Будете получать уведомления по {escape(dict(SYMBOLS).get(symbol, symbol))} "
                f"({escape(dict(PERIODS).get(period, period))}, {escape(ma_choice)})",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
            )

        elif data == "unsubscribe":
            unsubscribe(chat_id)
            await query.edit_message_text(
                "❌ Подписка отменена.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
            )

        # ---- Анализ тренда ----
        elif data == 'trend':
            keyboard = [[InlineKeyboardButton(escape(name), callback_data=f"symbol_{symbol}")] for symbol, name in SYMBOLS]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='menu')])
            await query.edit_message_text("Выберите криптовалюту:", reply_markup=InlineKeyboardMarkup(keyboard))

        elif data.startswith("symbol_"):
            symbol = data.split("_", 1)[1]
            context.user_data['symbol'] = symbol
            keyboard = [[InlineKeyboardButton(escape(name), callback_data=f"period_{period}")] for period, name in PERIODS]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='trend')])
            await query.edit_message_text("Выберите период анализа:", reply_markup=InlineKeyboardMarkup(keyboard))

        elif data.startswith("period_"):
            symbol = context.user_data.get('symbol', SYMBOLS[0][0])
            period = data.split("_", 1)[1]
            context.user_data['period'] = period
            keyboard = [[InlineKeyboardButton(escape(name), callback_data=f"ma_{ma}")] for ma, name in MA_CHOICES]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"symbol_{symbol}")])
            await query.edit_message_text("Выберите параметры скользящих MA:", reply_markup=InlineKeyboardMarkup(keyboard))

        elif data.startswith("ma_"):
            symbol = context.user_data.get('symbol', SYMBOLS[0][0])
            period = context.user_data.get('period', '1d')
            ma_choice = data.split("_", 1)[1]
            trend, recommendation, price, df, ma_s, ma_l = await asyncio.to_thread(get_trend, symbol, period, ma_choice)
            period_text = escape(dict(PERIODS).get(period, period))
            name = escape(dict(SYMBOLS).get(symbol, symbol))
            ma_label = f"MA{ma_s}/MA{ma_l}"

            message = (
                f'📊 <b>{name}</b> за период <b>{period_text}</b> ({ma_label}):\n'
                f'<b>Цена</b>: ${price}\n'
                f'<b>Тренд</b>: {escape(trend)}\n'
                f'<b>Рекомендация</b>: {escape(recommendation)}'
            )

            if df.empty or len(df) < max(ma_s, ma_l):
                await query.edit_message_text(
                    "Недостаточно данных для построения графика.",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='trend')]])
                )
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

        # ---- Актуальные цены ----
        elif data == 'prices':
            exchange = ccxt.binance()
            prices = []
            for symbol, name in SYMBOLS:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    prices.append(f"{escape(name)}: <b>${ticker['last']}</b>")
                except Exception:
                    prices.append(f"{escape(name)}: <i>Ошибка получения цены</i>")
            await query.edit_message_text(
                "📈 <b>Актуальные цены:</b>\n" + "\n".join(prices),
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
            )

        # ---- Помощь ----
        elif data == 'help':
            await query.edit_message_text(
                "ℹ️ <b>Инструкция:</b>\n"
                "1. Анализ тренда — пошагово выберите монету, период и параметры MA.\n"
                "2. Подписка на уведомления — выберите монету и MA для автоматических алертов.\n"
                "3. После анализа получите график, тренд и рекомендацию.\n"
                "4. Для возврата — кнопка Назад.",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]])
            )

        # ---- Личный кабинет ----
        elif data == 'profile':
            with app.app_context():
                user = User.query.filter_by(telegram_chat_id=str(chat_id)).first()
                if user:
                    sub_status = "✅ Подписан" if user.is_tg_subscribed else "❌ Не подписан"
                    symbol = escape(user.tg_symbol) if user.tg_symbol else "—"
                    ma = escape(user.tg_ma) if user.tg_ma else "—"
                    period = escape(getattr(user, "tg_period", "1d"))
                    name = escape(user.full_name) if getattr(user, "full_name", None) else "—"
                    msg = (
                        f"<b>👤 Личный кабинет</b>\n"
                        f"<b>Имя:</b> {name}\n"
                        f"<b>Статус подписки:</b> {sub_status}\n"
                        f"<b>Пара:</b> {symbol}\n"
                        f"<b>MA:</b> {ma}\n"
                        f"<b>Период:</b> {period}\n"
                        f"<b>Ваш chat_id:</b> <code>{escape(str(chat_id))}</code>"
                    )
                else:
                    msg = (
                        f"<b>👤 Личный кабинет</b>\n"
                        f"Вы ещё не зарегистрированы в системе.\n"
                        f"Ваш chat_id: <code>{escape(str(chat_id))}</code>"
                    )
                await query.edit_message_text(msg, parse_mode="HTML",
                                              reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data='menu')]]))
            return

        # ---- О боте ----
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

        # ---- Алерты ----
        elif data == "alerts":
            keyboard = [[InlineKeyboardButton(escape(name), callback_data=f"alert_symbol_{symbol}")] for symbol, name in SYMBOLS]
            keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data='menu')])
            await query.edit_message_text("Выберите криптовалюту для алерта:", reply_markup=InlineKeyboardMarkup(keyboard))

        elif data.startswith("alert_symbol_"):
            symbol = data.split("_", 2)[2]
            context.user_data['alert_symbol'] = symbol
            await query.edit_message_text(
                "Введите условие в формате:\n\n"
                f"<code>{escape('> 70000')}</code> — уведомить, если цена выше 70k\n"
                f"<code>{escape('< 60000')}</code> — уведомить, если цена ниже 60k",
                parse_mode="HTML"
            )
            context.user_data['awaiting_alert_price'] = True

    except BadRequest as e:
        logger.error(f"BadRequest in menu_callback: {e}")
        if "Message is not modified" not in str(e):
            raise



async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Используйте /start или /menu для вызова главного меню.\n'
        'Для справки напишите /help.'
    )

async def send_trend_notification_local(symbol, trend, recommendation, price, chat_id, period='1d', ma_choice="7,30"):
    name = dict(SYMBOLS).get(symbol, symbol)
    period_text = dict(PERIODS).get(period, period)
    ma_label = f"MA{ma_choice.replace(',', '/')}"

    message = (
        f"🔔 <b>Уведомление по подписке</b>\n"
        f"<b>{escape(name)}</b> ({escape(period_text)}, {escape(ma_label)})\n"
        f"<b>Тренд:</b> {escape(trend)}\n"
        f"<b>Рекомендация:</b> {escape(recommendation)}\n"
        f"<b>Цена:</b> ${price}"
    )
    try:
        await application.bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
        logger.info(f"Отправлено уведомление для {chat_id} по {symbol}")
    except Exception as e:
        logger.error(f"Не удалось отправить уведомление для {chat_id}: {e}")

async def notification_daemon():
    """Асинхронный демон для отправки уведомлений по подписке."""
    while True:
        logger.info("Проверка подписок для уведомлений...")
        try:
            def _get_subscribers_in_context():
                with app.app_context():
                    return get_all_subscribers()

            subscribers = await asyncio.to_thread(_get_subscribers_in_context)

            for user in subscribers:
                trend, recommendation, price, _, _, _ = await asyncio.to_thread(
                    get_trend, user.tg_symbol, user.tg_period, user.tg_ma
                )
                if 'Покупать' in recommendation or 'Продавать' in recommendation:
                    await send_trend_notification_local(
                        user.tg_symbol, trend, recommendation, price, user.telegram_chat_id, user.tg_period, user.tg_ma
                    )
        except Exception as e:
            logger.error(f"Ошибка в демоне уведомлений: {e}")
        await asyncio.sleep(3600)  # 1 час

async def run_async_tasks():
    """Запускает все асинхронные фоновые задачи."""
    await asyncio.gather(
        notification_daemon(),
        price_alert_daemon()
    )

async def price_alert_daemon():
    """Асинхронный демон для проверки ценовых алертов."""
    while True:
        try:
            def _get_alerts_and_commit_in_context():
                with app.app_context():
                    alerts = PriceAlert.query.filter_by(is_triggered=False).all()
                    if not alerts:
                        return None, None

                    symbols = list(set([a.symbol for a in alerts]))
                    exchange = ccxt.binance()
                    # Этот вызов сам по себе является блокирующим I/O, поэтому он внутри потока
                    tickers = exchange.fetch_tickers(symbols)

                    triggered_alerts = []
                    for alert in alerts:
                        current_price = tickers.get(alert.symbol, {}).get('last')
                        if not current_price:
                            continue

                        triggered = False
                        if alert.condition == '>' and current_price > alert.target_price:
                            triggered = True
                        elif alert.condition == '<' and current_price < alert.target_price:
                            triggered = True

                        if triggered:
                            alert.is_triggered = True
                            triggered_alerts.append((alert, current_price))

                    if triggered_alerts:
                        db.session.commit()
                    return triggered_alerts, None

            triggered_alerts, error = await asyncio.to_thread(_get_alerts_and_commit_in_context)

            if triggered_alerts:
                for alert, current_price in triggered_alerts:
                    message = (
                        f"🔔 Сработал алерт по {escape(alert.symbol)}!\n"
                        f"Условие: цена {escape(alert.condition)} ${alert.target_price}\n"
                        f"Текущая цена: <b>${current_price}</b>"
                    )
                    await application.bot.send_message(chat_id=alert.chat_id, text=message, parse_mode='HTML')

        except Exception as e:
            logger.error(f"Ошибка в демоне алертов: {e}")
        await asyncio.sleep(60)  # 1 минута


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('awaiting_alert_price'):
        chat_id = update.effective_chat.id
        symbol = context.user_data['alert_symbol']
        text = update.message.text.strip()
        if text.startswith(">") or text.startswith("<"):
            condition = text[0]
            try:
                target_price = float(text[1:].strip())
                with app.app_context():
                    alert = PriceAlert(chat_id=str(chat_id), symbol=symbol,
                                       condition=condition, target_price=target_price)
                    db.session.add(alert)
                    db.session.commit()
                await update.message.reply_text(
                    f"✅ Алерт создан: {symbol} {condition} {target_price}",
                    parse_mode="HTML"
                )
            except ValueError:
                await update.message.reply_text("❌ Ошибка: введите число после знака.")
        else:
            await update.message.reply_text("❌ Ошибка: используйте формат '> 70000' или '< 2000'.")
        context.user_data['awaiting_alert_price'] = False

async def run_async_tasks():
    """Запускает все асинхронные фоновые задачи."""
    await asyncio.gather(
        notification_daemon(),
        price_alert_daemon()
    )

def main():
    """Основная функция для запуска бота и фоновых задач."""
    # Добавление обработчиков
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Запускаем асинхронные задачи в отдельном потоке
    async_thread = Thread(target=lambda: asyncio.run(run_async_tasks()), daemon=True)
    async_thread.start()

    logger.info("Бот запущен и готов к работе...")
    # run_polling() является блокирующим и запускает свой собственный цикл asyncio
    application.run_polling()


if __name__ == '__main__':
    main()
