import React, { useState, useEffect } from 'react';
import CryptoCard from './CryptoCard';
import MarketTable from './MarketTable';
import AlertModal from './AlertModal';
import NotificationContainer from '../Common/NotificationContainer';
import ConnectionStatus from '../Common/ConnectionStatus';

const DashboardApp = () => {
  const [cryptoData, setCryptoData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAlertModal, setShowAlertModal] = useState(false);
  const [selectedCrypto, setSelectedCrypto] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [connectionError, setConnectionError] = useState(null);

  // Handle WebSocket connection and subscriptions
  useEffect(() => {
    loadCryptoData();
    
    // Setup WebSocket for real-time updates
    if (window.CryptoApp?.ws) {
      // Set up event listeners
      window.CryptoApp.ws.on('market_data_update', handleMarketDataUpdate);
      window.CryptoApp.ws.on('market_error', handleMarketError);
      window.CryptoApp.ws.on('connection', (data) => {
        setConnectionStatus(data.status);
        if (data.status === 'error') {
          setConnectionError(data.error || 'Connection error');
        } else {
          setConnectionError(null);
        }
        handleConnectionChange(data);
      });
      
      // Subscribe to market data
      window.CryptoApp.ws.subscribeToMarketData();
      
      // Connect WebSocket if not already connected
      if (!window.CryptoApp.ws.isConnected()) {
        console.log('Connecting to WebSocket...');
        window.CryptoApp.ws.connect();
      }
    } else {
      console.error('WebSocket service not available');
      setConnectionStatus('error');
      setConnectionError('WebSocket service not initialized');
    }

    // Auto-refresh every 60 seconds as fallback (reduced frequency due to WebSocket)
    const interval = setInterval(loadCryptoData, 60000);

    return () => {
      clearInterval(interval);
      if (window.CryptoApp?.ws) {
        window.CryptoApp.ws.off('market_data_update', handleMarketDataUpdate);
        window.CryptoApp.ws.off('market_error', handleMarketError);
        window.CryptoApp.ws.off('connection', handleConnectionChange);
      }
    };
  }, []);

  const loadCryptoData = async () => {
    try {
      setError(null);
      const data = await window.CryptoApp.api.getMarketData();
      setCryptoData(data);
      setLastUpdate(new Date());
    } catch (err) {
      setError('Ошибка загрузки данных рынка');
      console.error('Error loading crypto data:', err);
      
      // Show error notification
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.error('Не удалось загрузить данные рынка');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleMarketDataUpdate = (data) => {
    console.log('📊 Dashboard received market data update:', data.data?.length || 0, 'symbols');
    if (data.data && Array.isArray(data.data)) {
      setCryptoData(data.data);
      setLastUpdate(new Date());
      setError(null); // Clear any previous errors
      
      // Show success notification for real-time update
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.info('Данные рынка обновлены в реальном времени');
      }
    }
  };

  const handleMarketError = (data) => {
    console.error('❌ Market data error received:', data.error);
    const errorMsg = data.error || 'Unknown error';
    setError('Ошибка получения данных рынка: ' + errorMsg);
    setConnectionStatus('error');
    setConnectionError(errorMsg);
    
    if (window.CryptoApp?.notifications) {
      window.CryptoApp.notifications.error('Ошибка обновления данных рынка: ' + errorMsg);
    }
  };
  
  // Manual refresh function
  const handleManualRefresh = async () => {
    setLoading(true);
    setError(null);
    await loadCryptoData();
    
    // Try to reconnect WebSocket if not connected
    if (window.CryptoApp?.ws && !window.CryptoApp.ws.isConnected()) {
      window.CryptoApp.ws.connect();
    }
  };

  const handleConnectionChange = (data) => {
    console.log('🔌 WebSocket connection status:', data.status);
    
    if (data.status === 'connected') {
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.success('Подключение к real-time данным установлено');
      }
    } else if (data.status === 'disconnected') {
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.warning('Соединение с real-time данными потеряно');
      }
    } else if (data.status === 'error') {
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.error('Ошибка подключения к real-time данным');
      }
    }
  };

  const handleMarketUpdate = (data) => {
    // Legacy handler for backward compatibility
    setCryptoData(prevData => {
      return prevData.map(crypto => {
        const update = data.find(item => item.symbol === crypto.symbol);
        return update ? { ...crypto, ...update } : crypto;
      });
    });
    setLastUpdate(new Date());
  };

  const handleRefresh = () => {
    setLoading(true);
    loadCryptoData();
  };

  const handleAddToWatchlist = async (symbol) => {
    try {
      await window.CryptoApp.api.addToWatchlist(symbol);
      window.CryptoApp.notifications.success(`${symbol} добавлен в избранное`);
    } catch (err) {
      window.CryptoApp.notifications.error('Ошибка добавления в избранное');
    }
  };

  const handleSetAlert = (crypto) => {
    setSelectedCrypto(crypto);
    setShowAlertModal(true);
  };

  const handleCreateAlert = async (alertData) => {
    try {
      await window.CryptoApp.api.createAlert({
        symbol: selectedCrypto.symbol,
        ...alertData
      });
      
      window.CryptoApp.notifications.success(
        `Алерт создан для ${selectedCrypto.symbol}`,
        { title: 'Алерт создан' }
      );
      
      setShowAlertModal(false);
      setSelectedCrypto(null);
    } catch (err) {
      window.CryptoApp.notifications.error('Ошибка создания алерта');
    }
  };

  const formatLastUpdate = (date) => {
    return new Date(date).toLocaleTimeString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  if (loading && cryptoData.length === 0) {
    return (
      <div className="d-flex justify-content-center align-items-center" style={{ minHeight: '400px' }}>
        <div className="text-center">
          <div className="loading-spinner mb-3"></div>
          <p className="text-muted">Загрузка данных рынка...</p>
        </div>
      </div>
    );
  }

  if (error && cryptoData.length === 0) {
    return (
      <div className="alert alert-danger" role="alert">
        <h4 className="alert-heading">Ошибка загрузки</h4>
        <p>{error}</p>
        <button className="btn btn-outline-danger" onClick={handleRefresh}>
          <i className="bi bi-arrow-clockwise me-2"></i>
          Попробовать снова
        </button>
      </div>
    );
  }

  return (
    <div className="react-component">
      {/* Header with refresh button */}
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h1 className="h3 mb-0 text-gray-800">
            <i className="bi bi-graph-up-arrow text-primary"></i>
            Панель трейдера
          </h1>
          <p className="text-muted mb-0">Мониторинг криптовалютного рынка в реальном времени</p>
        </div>
        <div className="d-flex align-items-center">
          <ConnectionStatus />
          <div className="me-3 ms-3">
            <small className="text-muted">Последнее обновление:</small>
            <div className="fw-bold text-success">
              {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
          <button 
            className="btn btn-outline-primary btn-sm me-2" 
            onClick={handleRefresh}
            disabled={loading}
          >
            <i className={`bi bi-arrow-clockwise ${loading ? 'loading-spinner' : ''}`}></i>
            {loading ? ' Обновление...' : ' Обновить'}
          </button>
          <span className="badge bg-success">
            <i className="bi bi-circle-fill blink"></i> Live
          </span>
        </div>
      </div>

      {/* Crypto Cards Grid */}
      <div className="row mb-4">
        {cryptoData.slice(0, 4).map(crypto => (
          <div key={crypto.symbol} className="col-xl-3 col-md-6 mb-4">
            <CryptoCard
              crypto={crypto}
              onAddToWatchlist={handleAddToWatchlist}
              onSetAlert={handleSetAlert}
            />
          </div>
        ))}
      </div>

      {/* Market Data Table */}
      <div className="row">
        <div className="col-12">
          <MarketTable
            data={cryptoData}
            onAddToWatchlist={handleAddToWatchlist}
            onSetAlert={handleSetAlert}
            loading={loading}
          />
        </div>
      </div>

      {/* Alert Modal */}
      {showAlertModal && selectedCrypto && (
        <AlertModal
          crypto={selectedCrypto}
          onClose={() => {
            setShowAlertModal(false);
            setSelectedCrypto(null);
          }}
          onCreateAlert={handleCreateAlert}
        />
      )}

      {/* Notification Container */}
      <NotificationContainer />
    </div>
  );
};

export default DashboardApp;
