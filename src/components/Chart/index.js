import React from 'react';
import ReactDOM from 'react-dom/client';
import ChartApp from './ChartApp';

// Initialize React Chart App when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  const chartContainer = document.getElementById('react-chart');
  
  if (chartContainer) {
    const root = ReactDOM.createRoot(chartContainer);
    root.render(<ChartApp />);
  }
});
