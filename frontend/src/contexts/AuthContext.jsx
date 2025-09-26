import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is already logged in - PERSIST LOGIN ON REFRESH
    const authStatus = localStorage.getItem('isAuthenticated');
    const email = localStorage.getItem('userEmail');
    
    if (authStatus === 'true' && email) {
      setIsAuthenticated(true);
      setUserEmail(email);
    }
    setIsLoading(false);
  }, []);

  const login = (email) => {
    localStorage.setItem('isAuthenticated', 'true');
    localStorage.setItem('userEmail', email);
    setIsAuthenticated(true);
    setUserEmail(email);
  };

  const logout = () => {
    // Clear localStorage completely
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('userEmail');
    setIsAuthenticated(false);
    setUserEmail('');
    // Redirect to landing page after logout
    window.location.href = '/';
  };

  const value = {
    isAuthenticated,
    userEmail,
    isLoading,
    login,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
