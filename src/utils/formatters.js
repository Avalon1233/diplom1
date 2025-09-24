/**
 * Утилиты для форматирования данных в React компонентах
 */

// Форматирование цены в валюте
export const formatPrice = (price, currency = 'USD', options = {}) => {
  const defaultOptions = {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: price < 1 ? 4 : 2
  };

  return new Intl.NumberFormat('en-US', { ...defaultOptions, ...options }).format(price);
};

// Форматирование процентного изменения
export const formatChange = (change, showSign = true) => {
  const sign = showSign && change >= 0 ? '+' : '';
  return `${sign}${change.toFixed(2)}%`;
};

// Форматирование объема торгов
export const formatVolume = (volume) => {
  if (volume >= 1e12) {
    return `${(volume / 1e12).toFixed(1)}T`;
  } else if (volume >= 1e9) {
    return `${(volume / 1e9).toFixed(1)}B`;
  } else if (volume >= 1e6) {
    return `${(volume / 1e6).toFixed(1)}M`;
  } else if (volume >= 1e3) {
    return `${(volume / 1e3).toFixed(1)}K`;
  }
  return new Intl.NumberFormat('en-US').format(Math.round(volume));
};

// Форматирование больших чисел
export const formatNumber = (num, decimals = 0) => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(num);
};

// Форматирование времени
export const formatTime = (timestamp, options = {}) => {
  const defaultOptions = {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  };

  return new Date(timestamp).toLocaleTimeString('ru-RU', { ...defaultOptions, ...options });
};

// Форматирование даты
export const formatDate = (timestamp, options = {}) => {
  const defaultOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  };

  return new Date(timestamp).toLocaleDateString('ru-RU', { ...defaultOptions, ...options });
};

// Форматирование даты и времени
export const formatDateTime = (timestamp, options = {}) => {
  const defaultOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  };

  return new Date(timestamp).toLocaleString('ru-RU', { ...defaultOptions, ...options });
};

// Получение иконки криптовалюты
export const getCryptoIcon = (symbol) => {
  const iconMap = {
    'BTC': 'bi-currency-bitcoin text-warning',
    'ETH': 'bi-ethereum text-info',
    'BNB': 'bi-coin text-warning',
    'ADA': 'bi-coin text-primary',
    'SOL': 'bi-coin text-purple',
    'XRP': 'bi-coin text-info',
    'DOT': 'bi-coin text-danger',
    'DOGE': 'bi-coin text-warning',
    'AVAX': 'bi-coin text-danger',
    'MATIC': 'bi-coin text-purple'
  };

  const cryptoCode = symbol.split('-')[0].toUpperCase();
  return iconMap[cryptoCode] || 'bi-coin text-secondary';
};

// Получение полного названия криптовалюты
export const getCryptoName = (symbol) => {
  const nameMap = {
    'BTC': 'Bitcoin',
    'ETH': 'Ethereum',
    'BNB': 'Binance Coin',
    'ADA': 'Cardano',
    'SOL': 'Solana',
    'XRP': 'Ripple',
    'DOT': 'Polkadot',
    'DOGE': 'Dogecoin',
    'AVAX': 'Avalanche',
    'MATIC': 'Polygon'
  };

  const cryptoCode = symbol.split('-')[0].toUpperCase();
  return nameMap[cryptoCode] || cryptoCode;
};

// Форматирование символа криптовалюты для отображения
export const formatSymbol = (symbol) => {
  return symbol.replace('-', ' / ').toUpperCase();
};

// Определение цветового класса для изменения цены
export const getChangeColorClass = (change, prefix = 'text') => {
  if (change > 0) {
    return `${prefix}-success`;
  } else if (change < 0) {
    return `${prefix}-danger`;
  }
  return `${prefix}-muted`;
};

// Определение класса бейджа для изменения цены
export const getChangeBadgeClass = (change) => {
  if (change > 0) {
    return 'bg-success bg-opacity-10 text-success';
  } else if (change < 0) {
    return 'bg-danger bg-opacity-10 text-danger';
  }
  return 'bg-secondary bg-opacity-10 text-secondary';
};

// Определение иконки стрелки для изменения цены
export const getChangeArrowIcon = (change) => {
  if (change > 0) {
    return 'bi-arrow-up';
  } else if (change < 0) {
    return 'bi-arrow-down';
  }
  return 'bi-dash';
};

// Определение иконки треугольника для изменения цены
export const getChangeTriangleIcon = (change) => {
  return 'bi-triangle-fill';
};

// Получение стиля поворота треугольника
export const getTriangleRotation = (change) => {
  return change >= 0 ? 'rotate(0deg)' : 'rotate(180deg)';
};

// Вычисление ширины прогресс-бара на основе изменения
export const getProgressBarWidth = (change, multiplier = 5, maxWidth = 100) => {
  return Math.min(Math.abs(change) * multiplier, maxWidth);
};

// Форматирование рыночной капитализации
export const formatMarketCap = (marketCap) => {
  if (marketCap >= 1e12) {
    return `$${(marketCap / 1e12).toFixed(2)}T`;
  } else if (marketCap >= 1e9) {
    return `$${(marketCap / 1e9).toFixed(2)}B`;
  } else if (marketCap >= 1e6) {
    return `$${(marketCap / 1e6).toFixed(2)}M`;
  }
  return formatPrice(marketCap);
};

// Определение тренда на основе технических индикаторов
export const getTrendLabel = (trend) => {
  const trendMap = {
    'bullish': { label: 'Бычий', class: 'text-success', icon: 'bi-arrow-up' },
    'bearish': { label: 'Медвежий', class: 'text-danger', icon: 'bi-arrow-down' },
    'neutral': { label: 'Нейтральный', class: 'text-muted', icon: 'bi-dash' }
  };

  return trendMap[trend] || trendMap['neutral'];
};

// Определение рекомендации
export const getRecommendationLabel = (recommendation) => {
  const recommendationMap = {
    'buy': { label: 'Покупать', class: 'text-success', icon: 'bi-arrow-up-circle' },
    'sell': { label: 'Продавать', class: 'text-danger', icon: 'bi-arrow-down-circle' },
    'hold': { label: 'Держать', class: 'text-warning', icon: 'bi-pause-circle' }
  };

  return recommendationMap[recommendation] || recommendationMap['hold'];
};

// Сокращение длинных строк
export const truncateString = (str, maxLength = 20) => {
  if (str.length <= maxLength) return str;
  return str.substring(0, maxLength - 3) + '...';
};

// Валидация числовых значений
export const isValidNumber = (value) => {
  return !isNaN(value) && isFinite(value) && value !== null && value !== undefined;
};

// Безопасное форматирование числа
export const safeFormatNumber = (value, formatter, fallback = '--') => {
  if (!isValidNumber(value)) return fallback;
  try {
    return formatter(value);
  } catch (error) {
    console.warn('Error formatting number:', error);
    return fallback;
  }
};
