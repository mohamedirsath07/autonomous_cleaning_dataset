import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { GoogleOAuthProvider } from '@react-oauth/google';
import ErrorBoundary from './components/ErrorBoundary';
import './index.css'
import App from './App.jsx'

// Replace with your actual Google Client ID
const GOOGLE_CLIENT_ID = "832490195150-uvb0tj2qs9gsj6rkquhtgs8j6qug36kh.apps.googleusercontent.com";

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
        <App />
      </GoogleOAuthProvider>
    </ErrorBoundary>
  </StrictMode>,
)
