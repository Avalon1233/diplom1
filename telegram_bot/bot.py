# telegram_bot/bot.py
import requests
from telegram import Bot
from flask import current_app

def send_trend_notification(symbol, trend, recommendation, price, chat_id):
    bot_token = current_app.config.get('TELEGRAM_BOT_TOKEN')
    if not bot_token or not chat_id:
        print("TELEGRAM_BOT_TOKEN или chat_id не указаны")
        return
    text = (
        f"📈 Тренд для {symbol}:\n"
        f"Текущая цена: ${price}\n"
        f"Тренд: {trend}\n"
        f"Рекомендация: {recommendation}"
    )
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        resp = requests.post(url, data=data, timeout=10)
        print("Ответ Telegram:", resp.status_code, resp.text)
    except Exception as e:
        print(f"Ошибка Telegram: {e}")
