import React, { useState, useEffect } from 'react';

const ConnectionStatus = () => {
  const [connectionStatus, setConnectionStatus] = useState({
    isConnected: false,
    reconnectAttempts: 0,
    lastUpdate: null
  });

  useEffect(() => {
    const updateConnectionStatus = () => {
      if (window.CryptoApp?.ws) {
        const status = window.CryptoApp.ws.getConnectionStatus();
        setConnectionStatus({
          ...status,
          lastUpdate: new Date()
        });
      }
    };

    // Initial status check
    updateConnectionStatus();

    // Setup WebSocket event listeners
    if (window.CryptoApp?.ws) {
      const handleConnection = (data) => {
        updateConnectionStatus();
      };

      window.CryptoApp.ws.on('connection', handleConnection);
      window.CryptoApp.ws.on('connected', handleConnection);
      window.CryptoApp.ws.on('market_data_update', () => {
        setConnectionStatus(prev => ({
          ...prev,
          lastUpdate: new Date()
        }));
      });

      return () => {
        window.CryptoApp.ws.off('connection', handleConnection);
        window.CryptoApp.ws.off('connected', handleConnection);
      };
    }

    // Fallback: check connection status every 5 seconds
    const interval = setInterval(updateConnectionStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    if (connectionStatus.isConnected) {
      return 'success';
    } else if (connectionStatus.reconnectAttempts > 0) {
      return 'warning';
    } else {
      return 'danger';
    }
  };

  const getStatusText = () => {
    if (connectionStatus.isConnected) {
      return 'Подключено';
    } else if (connectionStatus.reconnectAttempts > 0) {
      return `Переподключение... (${connectionStatus.reconnectAttempts})`;
    } else {
      return 'Отключено';
    }
  };

  const getStatusIcon = () => {
    if (connectionStatus.isConnected) {
      return 'bi-wifi';
    } else if (connectionStatus.reconnectAttempts > 0) {
      return 'bi-arrow-clockwise';
    } else {
      return 'bi-wifi-off';
    }
  };

  return (
    <div className="d-flex align-items-center">
      <span className={`badge bg-${getStatusColor()} me-2`}>
        <i className={`bi ${getStatusIcon()} me-1`}></i>
        Real-time: {getStatusText()}
      </span>
      {connectionStatus.lastUpdate && (
        <small className="text-muted">
          Обновлено: {connectionStatus.lastUpdate.toLocaleTimeString()}
        </small>
      )}
    </div>
  );
};

export default ConnectionStatus;
