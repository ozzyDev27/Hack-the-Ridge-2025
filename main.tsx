/**
 * Application Entry Point
 * 
 * This file bootstraps the React application by mounting the root component
 * to the DOM. It includes React's StrictMode for additional development checks
 * and warnings.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/index.css';

// Mount the React application to the root DOM element
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
