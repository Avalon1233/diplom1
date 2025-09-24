import React from 'react';

const CryptoCard = ({ crypto, onAddToWatchlist, onSetAlert }) => {
  const isPositive = crypto.change >= 0;
  const changeClass = isPositive ? 'positive' : 'negative';
  const cardClass = isPositive ? 'price-up' : 'price-down';

  const getCryptoIcon = (symbol) => {
    if (symbol.includes('BTC')) return 'bi-currency-bitcoin text-warning';
    if (symbol.includes('ETH')) return 'bi-ethereum text-info';
    if (symbol.includes('BNB')) return 'bi-coin text-warning';
    if (symbol.includes('ADA')) return 'bi-coin text-primary';
    if (symbol.includes('SOL')) return 'bi-coin text-purple';
    return 'bi-coin text-secondary';
  };

  const getCryptoName = (symbol) => {
    if (symbol.includes('BTC')) return 'Bitcoin';
    if (symbol.includes('ETH')) return 'Ethereum';
    if (symbol.includes('BNB')) return 'Binance Coin';
    if (symbol.includes('ADA')) return 'Cardano';
    if (symbol.includes('SOL')) return 'Solana';
    return symbol.split('-')[0];
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: price < 1 ? 4 : 2
    }).format(price);
  };

  const formatChange = (change) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}%`;
  };

  return (
    <div className={`card crypto-price-card h-100 shadow-sm border-0 ${cardClass}`}>
      <div className="card-body">
        <div className="row no-gutters align-items-center">
          <div className="col mr-2">
            <div className="d-flex align-items-center mb-2">
              <div className="crypto-icon me-2">
                <i className={`bi ${getCryptoIcon(crypto.symbol)} fs-4`}></i>
              </div>
              <div>
                <div className="text-xs font-weight-bold text-uppercase mb-1">
                  {crypto.symbol.replace('-', '/')}
                </div>
                <small className="text-muted">{getCryptoName(crypto.symbol)}</small>
              </div>
            </div>
            
            <div className="h4 mb-0 font-weight-bold text-gray-800">
              {formatPrice(crypto.price)}
            </div>
            
            <div className="mt-2">
              <span className={`price-change ${changeClass} me-1`}>
                <i className={`bi ${isPositive ? 'bi-arrow-up' : 'bi-arrow-down'}`}></i>
                {formatChange(crypto.change)}
              </span>
              <small className="text-muted">24ч</small>
            </div>
            
            <div className="mt-2">
              <small className="text-muted">
                H: {formatPrice(crypto.high)} L: {formatPrice(crypto.low)}
              </small>
            </div>
          </div>
          
          <div className="col-auto">
            <div className="dropdown">
              <button 
                className="btn btn-sm btn-outline-secondary dropdown-toggle" 
                type="button" 
                data-bs-toggle="dropdown"
                aria-expanded="false"
              >
                <i className="bi bi-three-dots-vertical"></i>
              </button>
              <ul className="dropdown-menu">
                <li>
                  <a 
                    className="dropdown-item" 
                    href={`/trader/chart/${crypto.symbol}`}
                  >
                    <i className="bi bi-graph-up me-2"></i>График
                  </a>
                </li>
                <li>
                  <button 
                    className="dropdown-item" 
                    onClick={() => onAddToWatchlist(crypto.symbol)}
                  >
                    <i className="bi bi-star me-2"></i>В избранное
                  </button>
                </li>
                <li>
                  <button 
                    className="dropdown-item" 
                    onClick={() => onSetAlert(crypto)}
                  >
                    <i className="bi bi-bell me-2"></i>Установить алерт
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      <div className="card-footer bg-transparent border-0 pt-0">
        <div className="progress" style={{ height: '3px' }}>
          <div 
            className={`progress-bar ${isPositive ? 'bg-success' : 'bg-danger'}`}
            style={{ 
              width: `${Math.min(Math.abs(crypto.change) * 10, 100)}%` 
            }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default CryptoCard;
