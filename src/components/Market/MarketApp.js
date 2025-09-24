import React, { useState, useEffect } from 'react';
import MarketOverview from './MarketOverview';
import MarketTable from './MarketTable';
import AlertModal from '../Dashboard/AlertModal';
import NotificationContainer from '../Common/NotificationContainer';

const MarketApp = () => {
  const [marketData, setMarketData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCrypto, setSelectedCrypto] = useState(null);
  const [showAlertModal, setShowAlertModal] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadMarketData();
    
    // Setup WebSocket for real-time updates
    if (window.CryptoApp?.ws) {
      window.CryptoApp.ws.on('market_data_update', handleMarketDataUpdate);
      window.CryptoApp.ws.on('market_error', handleMarketError);
      window.CryptoApp.ws.on('connection', handleConnectionChange);
      window.CryptoApp.ws.on('price_update', handlePriceUpdate);
      window.CryptoApp.ws.subscribeToMarketData();
      
      // Connect WebSocket if not already connected
      if (!window.CryptoApp.ws.isConnected()) {
        window.CryptoApp.ws.connect();
      }
    }

    // Auto-refresh every 60 seconds as fallback
    const interval = setInterval(loadMarketData, 60000);

    return () => {
      clearInterval(interval);
      if (window.CryptoApp?.ws) {
        window.CryptoApp.ws.off('market_data_update', handleMarketDataUpdate);
        window.CryptoApp.ws.off('market_error', handleMarketError);
        window.CryptoApp.ws.off('connection', handleConnectionChange);
        window.CryptoApp.ws.off('price_update', handlePriceUpdate);
      }
    };
  }, []);

  const loadMarketData = async () => {
    try {
      setLoading(true);
      const data = await window.CryptoApp.api.getMarketData();
      setMarketData(data);
      setError(null);
    } catch (error) {
      console.error('Error loading market data:', error);
      setError('Ошибка загрузки данных рынка');
      
      // Show notification
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.error('Ошибка загрузки данных рынка');
      }
    } finally {
      setLoading(false);
    }
  };

const handleMarketDataUpdate = (data) => {
  console.log('📊 Market received market data update:', data.data?.length || 0, 'symbols');
  if (data.data && Array.isArray(data.data)) {
    setMarketData(data.data);
    setError(null);

    if (window.CryptoApp?.notifications) {
      window.CryptoApp.notifications.info('Рыночные данные обновлены в реальном времени');
    }
  }
};

const handleMarketError = (data) => {
  console.error('❌ Market data error received:', data.error);
  setError('Ошибка получения данных рынка: ' + data.error);

  if (window.CryptoApp?.notifications) {
    window.CryptoApp.notifications.error('Ошибка обновления данных рынка');
  }
};

const handleConnectionChange = (data) => {
  console.log('🔌 WebSocket connection status:', data.status);

  if (data.status === 'connected') {
    if (window.CryptoApp?.notifications) {
      window.CryptoApp.notifications.success('Real-time соединение установлено');
    }
  } else if (data.status === 'disconnected') {
    if (window.CryptoApp?.notifications) {
      window.CryptoApp.notifications.warning('Real-time соединение потеряно');
    }
  }
};

const handlePriceUpdate = (data) => {
  console.log('💰 Individual price update:', data.symbol, data.price);
  setMarketData(prevData => {
    return prevData.map(item => {
      if (item.symbol === data.symbol) {
        return {
          ...item,
          price: data.price,
          change_24h: data.change_24h || item.change_24h,
          volume_24h: data.volume || item.volume_24h,
          timestamp: data.timestamp
        };
      }
      return item;
    });
  });
};
  const handleAddToWatchlist = async (symbol) => {
    try {
      await window.CryptoApp.api.addToWatchlist(symbol);
      
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.success(`${symbol} добавлен в избранное`);
      }
    } catch (error) {
      console.error('Error adding to watchlist:', error);
      
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.error('Ошибка добавления в избранное');
      }
    }
  };

  const handleSetAlert = (crypto) => {
    setSelectedCrypto(crypto);
    setShowAlertModal(true);
  };

  const handleCreateAlert = async (alertData) => {
    try {
      await window.CryptoApp.api.createAlert(selectedCrypto.symbol, alertData);
      
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.success(
          `Алерт создан для ${selectedCrypto.symbol}`
        );
      }
      
      setShowAlertModal(false);
      setSelectedCrypto(null);
    } catch (error) {
      console.error('Error creating alert:', error);
      
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.error('Ошибка создания алерта');
      }
    }
  };

  const handleCloseAlert = () => {
    setShowAlertModal(false);
    setSelectedCrypto(null);
  };

  const handleRefresh = () => {
    loadMarketData();
  };

  const handleExport = () => {
    try {
      const csvContent = generateCSV(marketData);
      downloadCSV(csvContent, 'market_data.csv');
      
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.success('Данные экспортированы');
      }
    } catch (error) {
      console.error('Error exporting data:', error);
      
      if (window.CryptoApp?.notifications) {
        window.CryptoApp.notifications.error('Ошибка экспорта данных');
      }
    }
  };

  const generateCSV = (data) => {
    const headers = ['Symbol', 'Price', 'Change %', 'High 24h', 'Low 24h', 'Volume'];
    const rows = data.map(crypto => [
      crypto.symbol,
      crypto.price,
      crypto.change,
      crypto.high,
      crypto.low,
      crypto.volume
    ]);
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
  };

  const downloadCSV = (content, filename) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
      const url = URL.createObjectURL(blob);
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  if (error) {
    return (
      <div className="container-fluid">
        <div className="alert alert-danger" role="alert">
          <i className="bi bi-exclamation-triangle me-2"></i>
          {error}
          <button 
            className="btn btn-outline-danger btn-sm ms-3"
            onClick={handleRefresh}
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
      {/* Header Section */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="h3 mb-0 text-gray-800">
                <i className="bi bi-graph-up text-primary me-2"></i>
                Криптовалютный рынок
              </h1>
              <p className="text-muted mb-0">Полный обзор криптовалютного рынка</p>
            </div>
            <div>
              <button 
                className="btn btn-outline-primary btn-sm me-2" 
                onClick={handleRefresh}
                disabled={loading}
              >
                <i className="bi bi-arrow-clockwise me-1"></i>
                {loading ? 'Обновление...' : 'Обновить'}
              </button>
              <button 
                className="btn btn-outline-secondary btn-sm"
                onClick={handleExport}
              >
                <i className="bi bi-download me-1"></i>
                Экспорт
              </button>
              <span className="badge bg-success ms-2">
                <i className="bi bi-circle-fill blink"></i> Live
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Market Overview */}
      <MarketOverview data={marketData} loading={loading} />

      {/* Market Table */}
      <div className="row">
        <div className="col-12">
          <MarketTable
            data={marketData}
            onAddToWatchlist={handleAddToWatchlist}
            onSetAlert={handleSetAlert}
            onExport={handleExport}
            loading={loading}
          />
        </div>
      </div>

      {/* Alert Modal */}
      {showAlertModal && selectedCrypto && (
        <AlertModal
          crypto={selectedCrypto}
          onClose={handleCloseAlert}
          onCreateAlert={handleCreateAlert}
        />
      )}

      {/* Notifications */}
      <NotificationContainer />
    </div>
  );
};

export default MarketApp;
