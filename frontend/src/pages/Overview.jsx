import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Users, 
  TrendingUp, 
  MessageCircle,
  ArrowRight
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { systemAPI } from '../lib/api';
import toast from 'react-hot-toast';

const Overview = () => {
  const navigate = useNavigate();
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isBackendOnline, setIsBackendOnline] = useState(false);

  useEffect(() => {
    loadSystemHealth();
    
    // Set up periodic health checks every 30 seconds
    const healthCheckInterval = setInterval(loadSystemHealth, 30000);
    
    return () => clearInterval(healthCheckInterval);
  }, []);

  const loadSystemHealth = async () => {
    try {
      const healthResponse = await systemAPI.healthCheck();
      setSystemHealth(healthResponse);
      setIsBackendOnline(true);
    } catch (error) {
      console.error('Backend health check failed:', error);
      setIsBackendOnline(false);
      setSystemHealth({
        status: 'offline',
        database_status: 'disconnected',
        ml_model_status: 'unavailable'
      });
    } finally {
      setLoading(false);
    }
  };


  const QuickActionCard = ({ title, description, icon: Icon, color, onClick }) => (
    <div 
      className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20 p-6 cursor-pointer hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98]"
      onClick={onClick}
    >
      <div className="flex items-center space-x-4">
        <div className={`p-4 rounded-xl ${color} shadow-lg`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-gray-900 text-lg">{title}</h3>
          <p className="text-sm text-gray-600 mt-1">{description}</p>
        </div>
        <ArrowRight className="h-5 w-5 text-gray-400 group-hover:text-gray-600 transition-colors" />
      </div>
    </div>
  );

  const StatusCard = ({ title, status, isOnline }) => {
    const getStatusColor = () => {
      if (!isOnline) return 'bg-red-500';
      if (status === 'healthy' || status === 'connected' || status === 'ready') return 'bg-green-500';
      if (status === 'offline' || status === 'disconnected' || status === 'unavailable') return 'bg-red-500';
      return 'bg-yellow-500';
    };

    const getStatusText = () => {
      if (!isOnline) return 'offline';
      return status;
    };

    return (
      <div className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl border border-gray-200">
        <span className="text-sm font-semibold text-gray-700">{title}</span>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()} shadow-sm`}></div>
          <span className="text-sm text-gray-600 capitalize font-medium">{getStatusText()}</span>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span className="ml-3 text-gray-600">Loading...</span>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
            <TrendingUp className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
              Dashboard Overview
            </h1>
            <p className="text-gray-600">
              AI-powered lead classification and personalized recommendations
            </p>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20 p-6 mb-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-gray-900">System Status</h2>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isBackendOnline ? 'bg-green-500' : 'bg-red-500'} shadow-sm`}></div>
            <span className="text-sm text-gray-600 font-medium">
              {isBackendOnline ? 'Backend Online' : 'Backend Offline'}
            </span>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatusCard 
            title="Overall Health" 
            status={systemHealth?.status || 'healthy'} 
            isOnline={isBackendOnline}
          />
          <StatusCard 
            title="Database" 
            status={systemHealth?.database_status || 'connected'} 
            isOnline={isBackendOnline}
          />
          <StatusCard 
            title="ML Model" 
            status={systemHealth?.ml_model_status || 'ready'} 
            isOnline={isBackendOnline}
          />
        </div>
        {!isBackendOnline && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-700">
              ⚠️ Backend server is offline. Some features may not work properly. 
              Please check if the backend server is running.
            </p>
          </div>
        )}
      </div>


      {/* Quick Actions */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-gray-900 mb-6">Get Started</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <QuickActionCard
            title="Upload Leads"
            description="Upload CSV file to classify leads and get AI recommendations"
            icon={Upload}
            color="bg-gradient-to-r from-blue-500 to-blue-600"
            onClick={() => navigate('/upload')}
          />
          <QuickActionCard
            title="View Results"
            description="See classified leads with personalized recommendations"
            icon={Users}
            color="bg-gradient-to-r from-green-500 to-green-600"
            onClick={() => navigate('/leads')}
          />
        </div>
      </div>

      {/* How It Works */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-white/20 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-6">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
              <Upload className="h-8 w-8 text-white" />
            </div>
            <h3 className="font-bold text-gray-900 mb-2 text-lg">1. Upload CSV</h3>
            <p className="text-sm text-gray-600">Upload your leads data in CSV format</p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
              <TrendingUp className="h-8 w-8 text-white" />
            </div>
            <h3 className="font-bold text-gray-900 mb-2 text-lg">2. AI Classification</h3>
            <p className="text-sm text-gray-600">Leads are automatically classified as Hot, Warm, or Cold</p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-green-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
              <MessageCircle className="h-8 w-8 text-white" />
            </div>
            <h3 className="font-bold text-gray-900 mb-2 text-lg">3. Get Recommendations</h3>
            <p className="text-sm text-gray-600">Receive personalized messages for each lead</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Overview;
