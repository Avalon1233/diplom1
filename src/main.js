// Main JavaScript entry point for shared functionality
import 'bootstrap/dist/css/bootstrap.min.css';
import './styles/main.css';

// Global utilities and services
import { ApiService } from './services/ApiService';
import { WebSocketService } from './services/WebSocketService';
import { NotificationService } from './services/NotificationService';

// Initialize global services
window.CryptoApp = {
  api: new ApiService(),
  ws: new WebSocketService(),
  notifications: new NotificationService()
};

// Global event handlers
document.addEventListener('DOMContentLoaded', () => {
  console.log('Crypto Analysis System initialized');
  
  // Initialize WebSocket connection if user is authenticated
  if (document.body.dataset.userAuthenticated === 'true') {
    window.CryptoApp.ws.connect();
  }
});

// Export for use in other modules
export { ApiService, WebSocketService, NotificationService };
