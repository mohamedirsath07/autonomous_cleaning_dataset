import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { GoogleOAuthProvider } from '@react-oauth/google';
import ErrorBoundary from './components/ErrorBoundary';
import './index.css'
import App from './App.jsx'

// Replace with your actual Google Client ID
const GOOGLE_CLIENT_ID = "832490195150-igro2ni31839e7jbi0scc7cbh6u684p7.apps.googleusercontent.com";

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
        <App />
      </GoogleOAuthProvider>
    </ErrorBoundary>
  </StrictMode>,
)
