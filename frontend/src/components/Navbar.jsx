import React, { useState, useEffect } from 'react';
import { BarChart3, User, LogOut, ChevronDown, Wifi, WifiOff } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { systemAPI } from '../lib/api';

const Navbar = () => {
  const { userEmail, logout } = useAuth();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isBackendOnline, setIsBackendOnline] = useState(false);

  useEffect(() => {
    checkBackendStatus();
    
    // Check backend status every 30 seconds
    const statusInterval = setInterval(checkBackendStatus, 30000);
    
    return () => clearInterval(statusInterval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      await systemAPI.healthCheck();
      setIsBackendOnline(true);
    } catch (error) {
      setIsBackendOnline(false);
    }
  };

  const handleLogout = () => {
    logout();
    setIsDropdownOpen(false);
  };

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-primary-600 to-primary-700 rounded-lg flex items-center justify-center">
                <BarChart3 className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Lead HeatScore</h1>
                <p className="text-xs text-gray-500">AI-Powered Sales Lead Classification</p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 text-sm px-3 py-1.5 rounded-full ${
              isBackendOnline 
                ? 'text-green-600 bg-green-50' 
                : 'text-red-600 bg-red-50'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isBackendOnline ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <span className="font-medium">
                {isBackendOnline ? 'System Online' : 'System Offline'}
              </span>
            </div>
            
            {/* User Dropdown */}
            <div className="relative">
              <button 
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="flex items-center space-x-3 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all duration-200"
              >
                <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                  <User className="h-4 w-4 text-primary-600" />
                </div>
                <div className="text-left">
                  <div className="text-sm font-semibold text-gray-900">{userEmail}</div>
                  <div className="text-xs text-gray-500">Signed in</div>
                </div>
                <ChevronDown className="h-4 w-4" />
              </button>
              
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border border-gray-200 z-50 overflow-hidden">
                  <div className="py-2">
                    <div className="px-4 py-3 bg-gray-50 border-b border-gray-100">
                      <p className="font-semibold text-sm text-gray-900">{userEmail}</p>
                      <p className="text-xs text-gray-500">Active user</p>
                    </div>
                    <button
                      onClick={handleLogout}
                      className="flex items-center w-full px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <LogOut className="h-4 w-4 mr-3 text-gray-400" />
                      Sign out
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
