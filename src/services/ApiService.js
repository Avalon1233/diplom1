import axios from 'axios';

class ApiService {
  constructor() {
    this.baseURL = '/api';
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      }
    });

    // Add request interceptor for authentication
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Market data endpoints
  async getMarketData() {
    try {
      const response = await this.client.get('/react/market-data');
      return response.data;
    } catch (error) {
      console.error('Error fetching market data:', error);
      throw error;
    }
  }

  async getCryptoPrice(symbol) {
    try {
      const response = await this.client.get(`/crypto/${symbol}/price`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching price for ${symbol}:`, error);
      throw error;
    }
  }

  async getChartData(symbol, period = '1d') {
    try {
      const response = await this.client.get(`/crypto/${symbol}/chart`, {
        params: { period }
      });
      return response.data;
    } catch (error) {
      console.error(`Error fetching chart data for ${symbol}:`, error);
      throw error;
    }
  }

  // Alert management
  async createAlert(symbol, alertData) {
    try {
      const response = await this.client.post('/react/alerts', {
        symbol,
        ...alertData
      });
      return response.data;
    } catch (error) {
      console.error('Error creating alert:', error);
      throw error;
    }
  }

  async getAlerts() {
    try {
      const response = await this.client.get('/react/alerts');
      return response.data;
    } catch (error) {
      console.error('Error fetching alerts:', error);
      throw error;
    }
  }

  async deleteAlert(alertId) {
    try {
      const response = await this.client.delete('/react/alerts', {
        data: { alert_id: alertId }
      });
      return response.data;
    } catch (error) {
      console.error('Error deleting alert:', error);
      throw error;
    }
  }

  // Watchlist management
  async addToWatchlist(symbol) {
    try {
      const response = await this.client.post('/react/watchlist', { symbol });
      return response.data;
    } catch (error) {
      console.error('Error adding to watchlist:', error);
      throw error;
    }
  }

  async getWatchlist() {
    try {
      const response = await this.client.get('/react/watchlist');
      return response.data;
    } catch (error) {
      console.error('Error fetching watchlist:', error);
      throw error;
    }
  }

  async removeFromWatchlist(symbol) {
    try {
      const response = await this.client.delete('/react/watchlist', {
        data: { symbol }
      });
      return response.data;
    } catch (error) {
      console.error('Error removing from watchlist:', error);
      throw error;
    }
  }

  // Analysis endpoints
  async getTrendAnalysis(symbol, period = '1d') {
    try {
      const response = await this.client.get(`/react/analysis/${symbol}`, {
        params: { period }
      });
      return response.data;
    } catch (error) {
      console.error(`Error fetching trend analysis for ${symbol}:`, error);
      throw error;
    }
  }

  async getPortfolioAnalysis() {
    try {
      const response = await this.client.get('/analysis/portfolio');
      return response.data;
    } catch (error) {
      console.error('Error fetching portfolio analysis:', error);
      throw error;
    }
  }
}

export { ApiService };
