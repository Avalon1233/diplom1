# app/utils/security.py
"""
Security utilities for enhanced application security
"""
import secrets
import re
from urllib.parse import urlparse, urljoin
from flask import request


def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)


def generate_csrf_token():
    """Generate CSRF token"""
    return secrets.token_urlsafe(32)


def is_safe_url(target):
    """Check if URL is safe for redirect"""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in (
        'http', 'https') and ref_url.netloc == test_url.netloc


def sanitize_input(input_string, max_length=255):
    """Sanitize user input to prevent XSS and other attacks"""
    if not input_string:
        return ""

    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', str(input_string))

    # Limit length
    return sanitized[:max_length]


def get_client_ip():
    """Get real client IP address considering proxies"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr
