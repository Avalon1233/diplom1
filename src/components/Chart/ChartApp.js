import React, { useState, useEffect } from 'react';

const ChartApp = () => {
  const [symbol, setSymbol] = useState('BTC-USD');
  const [timeframe, setTimeframe] = useState('1h');
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Get symbol from URL path
    const pathParts = window.location.pathname.split('/');
    const urlSymbol = pathParts[pathParts.length - 1];
    if (urlSymbol && urlSymbol !== 'chart') {
      setSymbol(urlSymbol);
    }
    
    loadChartData();
  }, [symbol, timeframe]);

  const loadChartData = async () => {
    try {
      setLoading(true);
      
      // For now, we'll use the existing realtime prices API
      const response = await fetch(`/api/realtime_prices?symbol=${symbol}`);
      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setChartData(data);
      setError(null);
    } catch (error) {
      console.error('Error loading chart data:', error);
      setError('Ошибка загрузки данных графика');
    } finally {
      setLoading(false);
    }
  };

  const formatSymbolName = (symbol) => {
    if (symbol.includes('BTC')) return 'Bitcoin (BTC)';
    if (symbol.includes('ETH')) return 'Ethereum (ETH)';
    if (symbol.includes('BNB')) return 'Binance Coin (BNB)';
    if (symbol.includes('ADA')) return 'Cardano (ADA)';
    if (symbol.includes('SOL')) return 'Solana (SOL)';
    return symbol.replace('-', '/');
  };

  if (error) {
    return (
      <div className="container-fluid">
        <div className="alert alert-danger" role="alert">
          <i className="bi bi-exclamation-triangle me-2"></i>
          {error}
          <button 
            className="btn btn-outline-danger btn-sm ms-3"
            onClick={loadChartData}
          >
            <i className="bi bi-arrow-clockwise me-1"></i>
            Повторить
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="container-fluid">
      {/* Header */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="h3 mb-0 text-gray-800">
                <i className="bi bi-graph-up text-primary me-2"></i>
                График {formatSymbolName(symbol)}
              </h1>
              <p className="text-muted mb-0">Анализ ценовых движений в реальном времени</p>
            </div>
            <div>
              <div className="btn-group me-3" role="group">
                <button 
                  className={`btn btn-sm ${timeframe === '1m' ? 'btn-primary' : 'btn-outline-primary'}`}
                  onClick={() => setTimeframe('1m')}
                >
                  1м
                </button>
                <button 
                  className={`btn btn-sm ${timeframe === '5m' ? 'btn-primary' : 'btn-outline-primary'}`}
                  onClick={() => setTimeframe('5m')}
                >
                  5м
                </button>
                <button 
                  className={`btn btn-sm ${timeframe === '1h' ? 'btn-primary' : 'btn-outline-primary'}`}
                  onClick={() => setTimeframe('1h')}
                >
                  1ч
                </button>
                <button 
                  className={`btn btn-sm ${timeframe === '1d' ? 'btn-primary' : 'btn-outline-primary'}`}
                  onClick={() => setTimeframe('1d')}
                >
                  1д
                </button>
              </div>
              <button 
                className="btn btn-outline-primary btn-sm"
                onClick={loadChartData}
                disabled={loading}
              >
                <i className="bi bi-arrow-clockwise me-1"></i>
                {loading ? 'Обновление...' : 'Обновить'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div className="row">
        <div className="col-12">
          <div className="card shadow-sm border-0">
            <div className="card-body">
              {loading ? (
                <div className="d-flex justify-content-center align-items-center" style={{ height: '400px' }}>
                  <div className="loading-spinner"></div>
                  <span className="ms-2">Загрузка данных...</span>
                </div>
              ) : chartData ? (
                <div className="chart-container" style={{ height: '400px' }}>
                  {/* Здесь будет интегрирован Chart.js или другая библиотека графиков */}
                  <div className="d-flex justify-content-center align-items-center h-100">
                    <div className="text-center">
                      <i className="bi bi-graph-up display-1 text-muted"></i>
                      <h5 className="mt-3 text-muted">График будет добавлен в следующих обновлениях</h5>
                      <p className="text-muted">
                        Данные загружены: {chartData.timestamps?.length || 0} точек
                      </p>
                      {chartData.prices && chartData.prices.length > 0 && (
                        <div className="mt-3">
                          <span className="badge bg-primary fs-6">
                            Последняя цена: ${chartData.prices[chartData.prices.length - 1]}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="d-flex justify-content-center align-items-center" style={{ height: '400px' }}>
                  <div className="text-center">
                    <i className="bi bi-exclamation-circle display-1 text-warning"></i>
                    <h5 className="mt-3 text-muted">Нет данных для отображения</h5>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Chart Controls */}
      <div className="row mt-4">
        <div className="col-md-6">
          <div className="card shadow-sm border-0">
            <div className="card-header bg-white border-0">
              <h6 className="mb-0">
                <i className="bi bi-gear me-2"></i>
                Настройки графика
              </h6>
            </div>
            <div className="card-body">
              <div className="mb-3">
                <label className="form-label">Криптовалюта</label>
                <select 
                  className="form-select"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                >
                  <option value="BTC-USD">Bitcoin (BTC/USD)</option>
                  <option value="ETH-USD">Ethereum (ETH/USD)</option>
                  <option value="BNB-USD">Binance Coin (BNB/USD)</option>
                  <option value="ADA-USD">Cardano (ADA/USD)</option>
                  <option value="SOL-USD">Solana (SOL/USD)</option>
                </select>
              </div>
              <div className="mb-3">
                <label className="form-label">Временной интервал</label>
                <select 
                  className="form-select"
                  value={timeframe}
                  onChange={(e) => setTimeframe(e.target.value)}
                >
                  <option value="1m">1 минута</option>
                  <option value="5m">5 минут</option>
                  <option value="15m">15 минут</option>
                  <option value="1h">1 час</option>
                  <option value="4h">4 часа</option>
                  <option value="1d">1 день</option>
                </select>
              </div>
            </div>
          </div>
        </div>
        
        <div className="col-md-6">
          <div className="card shadow-sm border-0">
            <div className="card-header bg-white border-0">
              <h6 className="mb-0">
                <i className="bi bi-info-circle me-2"></i>
                Информация
              </h6>
            </div>
            <div className="card-body">
              <div className="row">
                <div className="col-6">
                  <div className="text-center">
                    <div className="text-muted small">Текущая цена</div>
                    <div className="h5 mb-0">
                      {chartData?.prices?.length > 0 
                        ? `$${chartData.prices[chartData.prices.length - 1]}`
                        : '--'
                      }
                    </div>
                  </div>
                </div>
                <div className="col-6">
                  <div className="text-center">
                    <div className="text-muted small">Точек данных</div>
                    <div className="h5 mb-0">
                      {chartData?.timestamps?.length || 0}
                    </div>
                  </div>
                </div>
              </div>
              <div className="mt-3">
                <small className="text-muted">
                  Последнее обновление: {new Date().toLocaleTimeString()}
                </small>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartApp;
