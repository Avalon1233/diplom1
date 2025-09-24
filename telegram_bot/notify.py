import os
import requests
from config import Config

def send_trend_notification(symbol, trend, recommendation, price, chat_id, period='1d', ma_choice="7,30"):
    token = Config.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в окружении")
    ma_label = f"MA{ma_choice.replace(',', '/MA')}"
    text = (
        f"📊 Тренд по {symbol} ({ma_label}):\n"
        f"Цена: ${price}\n"
        f"Тренд: {trend}\n"
        f"Рекомендация: {recommendation}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    response = requests.post(url, json=payload)
    if not response.ok:
        raise RuntimeError(f"Telegram API error: {response.text}")

#def test():
