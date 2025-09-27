import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

// Context
import { AuthProvider, useAuth } from './contexts/AuthContext';

// Components
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import ProtectedRoute from './components/ProtectedRoute';

// Pages
import Login from './pages/Login';
import Landing from './pages/Landing';
import Upload from './pages/Upload';
import Leads from './pages/Leads';
import Overview from './pages/Overview';
import Detail from './pages/Detail';

// Component to handle root route - ALWAYS show Landing page first
const RootRoute = () => {
  return <Landing />;
};

function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<RootRoute />} />
            <Route path="/landing" element={<Landing />} />
            <Route path="/login" element={<Login />} />
            
            {/* Protected Routes */}
            <Route path="/*" element={
              <ProtectedRoute>

                <Navbar />
                <div className="flex">
                  <Sidebar />
                  <main className="flex-1 p-6">
                    <Routes>
                      <Route path="/dashboard" element={<Overview />} />
                      <Route path="/upload" element={<Upload />} />
                      <Route path="/leads" element={<Leads />} />
                      <Route path="/leads/:id" element={<Detail />} />
                    </Routes>
                  </main>
                </div>
              </ProtectedRoute>
            } />
          </Routes>
          
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;
