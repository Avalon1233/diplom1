import React from 'react';
import ReactDOM from 'react-dom/client';
import MarketApp from './MarketApp';

// Initialize React Market App when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  const marketContainer = document.getElementById('react-market');
  
  if (marketContainer) {
    const root = ReactDOM.createRoot(marketContainer);
    root.render(<MarketApp />);
  }
});
