import { io } from 'socket.io-client';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.listeners = new Map();
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
  }

  connect() {
    if (this.socket?.connected) {
      console.log('WebSocket already connected');
      return;
    }

    // Close existing connection if it exists
    if (this.socket) {
      this.socket.close();
    }

    // Initialize new socket connection with timeout and reconnection settings
    this.socket = io({
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      upgrade: true,
      rememberUpgrade: true,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 10000
    });

    this.setupEventHandlers();
  }

  setupEventHandlers() {
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection', { status: 'connected' });
      
      // Resubscribe to market data after reconnection
      if (this.subscribedMarket) {
        this.subscribeToMarketData();
      }
      
      // Resubscribe to any price subscriptions
      if (this.priceSubscriptions && this.priceSubscriptions.size > 0) {
        this.priceSubscriptions.forEach(symbol => {
          this.subscribeToPrice(symbol);
        });
      }
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.emit('connection', { status: 'disconnected', reason });
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, don't reconnect
        return;
      }
      
      this.handleReconnect();
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.emit('connection', { status: 'error', error });
      this.handleReconnect();
    });

    // Market data events
    this.socket.on('market_data_update', (data) => {
      console.log('📊 Received market data update:', data.data?.length || 0, 'symbols');
      this.emit('market_data_update', data);
    });

    this.socket.on('market_update', (data) => {
      this.emit('market_update', data);
    });

    this.socket.on('price_update', (data) => {
      console.log('💰 Price update:', data.symbol, data.price);
      this.emit('price_update', data);
    });

    this.socket.on('market_error', (data) => {
      console.error('❌ Market data error:', data.error);
      this.emit('market_error', data);
    });

    this.socket.on('alert_triggered', (data) => {
      this.emit('alert_triggered', data);
    });

    // Connection confirmation events
    this.socket.on('connected', (data) => {
      console.log('📡 WebSocket connection confirmed:', data.message);
      this.emit('connected', data);
    });

    this.socket.on('subscribed_market', (data) => {
      console.log('✅ Subscribed to market data:', data.message);
      this.emit('subscribed_market', data);
    });

    this.socket.on('subscribed', (data) => {
      console.log('✅ Subscribed to price updates:', data.symbol);
      this.emit('subscribed', data);
    });

    this.socket.on('unsubscribed', (data) => {
      console.log('❌ Unsubscribed from price updates:', data.symbol);
      this.emit('unsubscribed', data);
    });

    // Portfolio events
    this.socket.on('portfolio_update', (data) => {
      this.emit('portfolio_update', data);
    });

    // Notification events
    this.socket.on('notification', (data) => {
      this.emit('notification', data);
    });
  }

  handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
      
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);
      
      this.reconnectTimeout = setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('connection', { 
        status: 'error', 
        error: 'Failed to reconnect after maximum attempts',
        timestamp: new Date().toISOString()
      });
      
      // Reset reconnection attempts after a delay to allow manual retry
      setTimeout(() => {
        this.reconnectAttempts = 0;
      }, 60000);
    }
  }

  unsubscribeFromCrypto(symbol) {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe_price', { symbol });
    }
  }

  // Subscribe to market updates
  subscribeToMarket() {
    if (this.socket?.connected) {
      this.socket.emit('subscribe_market');
    }
  }

  unsubscribeFromMarket() {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe_market');
    }
  }

  // New methods for enhanced WebSocket functionality
  subscribeToMarketData() {
    if (this.socket?.connected) {
      this.socket.emit('subscribe_market');
    } else {
      console.warn('WebSocket not connected, cannot subscribe to market data');
    }
  }

  subscribeToPriceUpdates(symbol) {
    if (this.socket?.connected) {
      this.socket.emit('subscribe_price', { symbol });
    } else {
      console.warn('WebSocket not connected, cannot subscribe to price updates for', symbol);
    }
  }

  unsubscribeFromPriceUpdates(symbol) {
    if (this.socket?.connected) {
      this.socket.emit('unsubscribe_price', { symbol });
    }
  }

  // Get connection status
  getConnectionStatus() {
    return {
      isConnected: this.socket?.connected || false,
      reconnectAttempts: this.reconnectAttempts,
      maxReconnectAttempts: this.maxReconnectAttempts
    };
  }

  // Event listener management
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${event}:`, error);
        }
      });
    }
  }

  // Send custom events to server
  send(event, data) {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket not connected, cannot send event:', event);
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.listeners.clear();
  }

  isConnected() {
    return this.socket?.connected || false;
  }
}

export { WebSocketService };
