import React from 'react';
import { createRoot } from 'react-dom/client';
import DashboardApp from './DashboardApp';

const container = document.getElementById('react-dashboard');

if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <DashboardApp />
    </React.StrictMode>
  );
} else {
  console.error('Failed to find the root element for the React dashboard.');
}
