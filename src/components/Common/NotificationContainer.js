import React, { useState, useEffect } from 'react';

const NotificationContainer = () => {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    if (!window.CryptoApp?.notifications) return;

    const handleNotificationEvent = (event, data) => {
      switch (event) {
        case 'add':
          setNotifications(prev => [data, ...prev.slice(0, 4)]); // Keep only 5 latest
          break;
        case 'remove':
          setNotifications(prev => prev.filter(n => n.id !== data.id));
          break;
        case 'clear':
          setNotifications([]);
          break;
        default:
          break;
      }
    };

    window.CryptoApp.notifications.addListener(handleNotificationEvent);

    // Load existing notifications
    const existing = window.CryptoApp.notifications.getAll().slice(0, 5);
    setNotifications(existing);

    return () => {
      window.CryptoApp.notifications.removeListener(handleNotificationEvent);
    };
  }, []);

  const handleClose = (id) => {
    window.CryptoApp.notifications.remove(id);
  };

  const handleAction = (action) => {
    if (window.CryptoApp.notifications.executeAction) {
      window.CryptoApp.notifications.executeAction(action);
    }
  };

  if (notifications.length === 0) return null;

  return (
    <div className="notification-container">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={`notification ${notification.type || 'info'}`}
        >
          <div className="notification-header">
            <h6 className="notification-title">
              {notification.icon && (
                <i className={`bi ${notification.icon} me-2`}></i>
              )}
              {notification.title}
            </h6>
            <button
              className="notification-close"
              onClick={() => handleClose(notification.id)}
            >
              <i className="bi bi-x"></i>
            </button>
          </div>
          
          <div className="notification-body">
            {notification.message}
          </div>

          {notification.actions && notification.actions.length > 0 && (
            <div className="notification-actions">
              {notification.actions.map((action, index) => (
                <button
                  key={index}
                  className="notification-action"
                  onClick={() => handleAction(action)}
                >
                  {action.label}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default NotificationContainer;
