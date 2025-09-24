# app/utils/validators.py
"""
Enhanced validation utilities for user input and data validation
"""
import re
from typing import List, Optional
from email_validator import validate_email, EmailNotValidError


def validate_password_strength(password: str) -> List[str]:
    """
    Validate password strength and return list of errors

    Requirements:
    - At least 8 characters long
    - Contains uppercase letter
    - Contains lowercase letter
    - Contains digit
    - Contains special character
    """
    errors = []

    if len(password) < 8:
        errors.append("Пароль должен содержать минимум 8 символов")

    if not re.search(r'[A-Z]', password):
        errors.append("Пароль должен содержать хотя бы одну заглавную букву")

    if not re.search(r'[a-z]', password):
        errors.append("Пароль должен содержать хотя бы одну строчную букву")

    if not re.search(r'\d', password):
        errors.append("Пароль должен содержать хотя бы одну цифру")

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append(
            "Пароль должен содержать хотя бы один специальный символ")

    # Check for common weak passwords
    weak_passwords = [
        'password', '123456', 'qwerty', 'admin', 'letmein',
        'welcome', 'monkey', '1234567890', 'password123'
    ]

    if password.lower() in weak_passwords:
        errors.append("Пароль слишком простой, выберите более сложный")

    return errors


def validate_username(username: str) -> List[str]:
    """Validate username format and requirements"""
    errors = []

    if not username:
        errors.append("Имя пользователя обязательно")
        return errors

    if len(username) < 3:
        errors.append("Имя пользователя должно содержать минимум 3 символа")

    if len(username) > 30:
        errors.append("Имя пользователя не должно превышать 30 символов")

    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        errors.append(
            "Имя пользователя может содержать только буквы, цифры, дефисы и подчеркивания")

    if username.startswith('_') or username.startswith('-'):
        errors.append("Имя пользователя не может начинаться с символа _ или -")

    return errors


def validate_email_address(
        email: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Validate email address
    Returns: (is_valid, normalized_email, error_message)
    """
    try:
        valid = validate_email(email)
        return True, valid.email, None
    except EmailNotValidError as e:
        return False, None, str(e)


def validate_crypto_symbol(symbol: str) -> List[str]:
    """Validate cryptocurrency symbol format"""
    errors = []

    if not symbol:
        errors.append("Символ криптовалюты обязателен")
        return errors

    # Remove common separators and normalize
    symbol = symbol.upper().replace('/', '').replace('-', '')

    if len(symbol) < 2 or len(symbol) > 10:
        errors.append("Символ должен содержать от 2 до 10 символов")

    if not re.match(r'^[A-Z0-9]+$', symbol):
        errors.append("Символ может содержать только заглавные буквы и цифры")

    return errors


def validate_price(price: str) -> tuple[bool, Optional[float], Optional[str]]:
    """
    Validate price input
    Returns: (is_valid, parsed_price, error_message)
    """
    try:
        price_float = float(price)
        if price_float <= 0:
            return False, None, "Цена должна быть положительным числом"
        if price_float > 1000000000:  # 1 billion limit
            return False, None, "Цена слишком большая"
        return True, price_float, None
    except (ValueError, TypeError):
        return False, None, "Неверный формат цены"


def validate_telegram_chat_id(chat_id: str) -> List[str]:
    """Validate Telegram chat ID format"""
    errors = []

    if not chat_id:
        return errors  # Optional field

    # Remove whitespace
    chat_id = chat_id.strip()

    # Telegram chat IDs are typically numeric (can be negative for groups)
    if not re.match(r'^-?\d+$', chat_id):
        errors.append(
            "Chat ID должен содержать только цифры (может начинаться с минуса)")

    if len(chat_id) > 20:
        errors.append("Chat ID слишком длинный")

    return errors


def validate_timeframe(timeframe: str) -> bool:
    """Validate trading timeframe format"""
    valid_timeframes = [
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    ]
    return timeframe in valid_timeframes


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit(
            '.', 1) if '.' in filename else (
            filename, '')
        filename = name[:250] + ('.' + ext if ext else '')

    return filename or 'unnamed'


def validate_json_data(data: dict, required_fields: List[str]) -> List[str]:
    """Validate JSON data contains required fields"""
    errors = []

    for field in required_fields:
        if field not in data:
            errors.append(f"Поле '{field}' обязательно")
        elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            errors.append(f"Поле '{field}' не может быть пустым")

    return errors
