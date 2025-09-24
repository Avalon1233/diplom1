class NotificationService {
  constructor() {
    this.notifications = [];
    this.listeners = new Set();
    this.maxNotifications = 50;
    
    // Request notification permission on initialization
    this.requestPermission();
  }

  async requestPermission() {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission();
      console.log('Notification permission:', permission);
      return permission === 'granted';
    }
    return false;
  }

  // Add notification to the queue
  add(notification) {
    const id = Date.now() + Math.random();
    const notificationObj = {
      id,
      timestamp: new Date(),
      read: false,
      ...notification
    };

    this.notifications.unshift(notificationObj);
    
    // Keep only the latest notifications
    if (this.notifications.length > this.maxNotifications) {
      this.notifications = this.notifications.slice(0, this.maxNotifications);
    }

    // Notify listeners
    this.notifyListeners('add', notificationObj);

    // Show browser notification if permitted
    this.showBrowserNotification(notificationObj);

    return id;
  }

  // Show different types of notifications
  success(message, options = {}) {
    return this.add({
      type: 'success',
      title: options.title || 'Успех',
      message,
      icon: 'bi-check-circle',
      ...options
    });
  }

  error(message, options = {}) {
    return this.add({
      type: 'error',
      title: options.title || 'Ошибка',
      message,
      icon: 'bi-exclamation-triangle',
      persistent: true,
      ...options
    });
  }

  warning(message, options = {}) {
    return this.add({
      type: 'warning',
      title: options.title || 'Предупреждение',
      message,
      icon: 'bi-exclamation-circle',
      ...options
    });
  }

  info(message, options = {}) {
    return this.add({
      type: 'info',
      title: options.title || 'Информация',
      message,
      icon: 'bi-info-circle',
      ...options
    });
  }

  // Trading specific notifications
  priceAlert(symbol, currentPrice, targetPrice, condition) {
    return this.add({
      type: 'alert',
      title: 'Ценовой алерт',
      message: `${symbol}: цена ${condition} $${targetPrice} (текущая: $${currentPrice})`,
      icon: 'bi-bell',
      persistent: true,
      actions: [
        { label: 'Посмотреть график', action: 'view_chart', data: { symbol } },
        { label: 'Создать ордер', action: 'create_order', data: { symbol, price: currentPrice } }
      ]
    });
  }

  trendChange(symbol, oldTrend, newTrend, confidence) {
    const trendIcon = newTrend === 'bullish' ? 'bi-arrow-up' : 'bi-arrow-down';
    const trendColor = newTrend === 'bullish' ? 'success' : 'danger';
    
    return this.add({
      type: trendColor,
      title: 'Изменение тренда',
      message: `${symbol}: тренд изменился с ${oldTrend} на ${newTrend} (уверенность: ${confidence}%)`,
      icon: trendIcon,
      actions: [
        { label: 'Анализ', action: 'view_analysis', data: { symbol } }
      ]
    });
  }

  portfolioUpdate(change, percentage) {
    const type = change >= 0 ? 'success' : 'danger';
    const icon = change >= 0 ? 'bi-arrow-up' : 'bi-arrow-down';
    const sign = change >= 0 ? '+' : '';
    
    return this.add({
      type,
      title: 'Обновление портфеля',
      message: `Изменение: ${sign}${change.toFixed(2)}% (${sign}${percentage.toFixed(2)}%)`,
      icon
    });
  }

  // Browser notification
  showBrowserNotification(notification) {
    if ('Notification' in window && Notification.permission === 'granted') {
      const browserNotification = new Notification(notification.title, {
        body: notification.message,
        icon: '/static/images/crypto-icon.png',
        tag: notification.id,
        requireInteraction: notification.persistent || false
      });

      browserNotification.onclick = () => {
        window.focus();
        this.handleNotificationClick(notification);
        browserNotification.close();
      };

      // Auto close non-persistent notifications
      if (!notification.persistent) {
        setTimeout(() => {
          browserNotification.close();
        }, 5000);
      }
    }
  }

  // Handle notification actions
  handleNotificationClick(notification) {
    this.markAsRead(notification.id);
    
    if (notification.actions && notification.actions.length > 0) {
      // For now, just execute the first action
      const action = notification.actions[0];
      this.executeAction(action);
    }
  }

  executeAction(action) {
    switch (action.action) {
      case 'view_chart':
        window.location.href = `/trader/chart/${action.data.symbol}`;
        break;
      case 'view_analysis':
        window.location.href = `/analyst/analyze?symbol=${action.data.symbol}`;
        break;
      case 'create_order':
        // Open order modal or redirect to trading page
        console.log('Create order action:', action.data);
        break;
      default:
        console.log('Unknown action:', action);
    }
  }

  // Mark notification as read
  markAsRead(id) {
    const notification = this.notifications.find(n => n.id === id);
    if (notification && !notification.read) {
      notification.read = true;
      this.notifyListeners('read', notification);
    }
  }

  // Mark all notifications as read
  markAllAsRead() {
    let changed = false;
    this.notifications.forEach(notification => {
      if (!notification.read) {
        notification.read = true;
        changed = true;
      }
    });
    
    if (changed) {
      this.notifyListeners('read_all');
    }
  }

  // Remove notification
  remove(id) {
    const index = this.notifications.findIndex(n => n.id === id);
    if (index !== -1) {
      const removed = this.notifications.splice(index, 1)[0];
      this.notifyListeners('remove', removed);
      return removed;
    }
    return null;
  }

  // Clear all notifications
  clear() {
    this.notifications = [];
    this.notifyListeners('clear');
  }

  // Get notifications
  getAll() {
    return [...this.notifications];
  }

  getUnread() {
    return this.notifications.filter(n => !n.read);
  }

  getUnreadCount() {
    return this.getUnread().length;
  }

  // Event listeners
  addListener(callback) {
    this.listeners.add(callback);
  }

  removeListener(callback) {
    this.listeners.delete(callback);
  }

  notifyListeners(event, data) {
    this.listeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('Error in notification listener:', error);
      }
    });
  }
}

export { NotificationService };
