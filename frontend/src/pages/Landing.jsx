import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { 
  Shield, 
  Zap, 
  Users, 
  Upload, 
  Database, 
  ShieldCheck,
  ArrowRight,
  BarChart3,
  Target,
  Mail,
  MessageSquare,
  TrendingUp
} from 'lucide-react';

const Landing = () => {
  const { isAuthenticated, userEmail, logout, isLoading } = useAuth();

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-4">
            <Target className="h-6 w-6 text-white animate-pulse" />
          </div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  const handleGetStarted = () => {
    if (isAuthenticated) {
      // If already logged in, go to dashboard
      window.location.href = '/dashboard';
    } else {
      // If not logged in, go to login
      window.location.href = '/login';
    }
  };

  const handleSignIn = () => {
    window.location.href = '/login';
  };

  const handleViewDemo = () => {
    if (isAuthenticated) {
      window.location.href = '/metrics';
    } else {
      window.location.href = '/login';
    }
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Target className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900">Lead HeatScore</span>
            </div>
            <div className="flex items-center space-x-4">
              {isAuthenticated ? (
                <>
                  <span className="text-gray-600">Welcome, {userEmail}</span>
                  <button 
                    onClick={() => window.location.href = '/dashboard'}
                    className="text-gray-600 hover:text-gray-900 px-3 py-2"
                  >
                    Dashboard
                  </button>
                  <button 
                    onClick={logout}
                    className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    Logout
                  </button>
                </>
              ) : (
                <>
                  <button 
                    onClick={handleSignIn}
                    className="text-gray-600 hover:text-gray-900 px-3 py-2"
                  >
                    Sign In
                  </button>
                  <button 
                    onClick={handleGetStarted}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Get Started
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-50 to-indigo-100 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Lead Classification
            <span className="text-blue-600"> Made Simple</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            AI-powered lead scoring with advanced ML models, personalized recommendations, and comprehensive analytics. 
            Transform your lead management with intelligent heat scoring and automated next actions.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={handleGetStarted}
              className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center"
            >
              {isAuthenticated ? 'Go to Dashboard' : 'Get Started'}
              <ArrowRight className="ml-2 h-5 w-5" />
            </button>
            <button 
              onClick={isAuthenticated ? handleViewDemo : handleSignIn}
              className="bg-white text-gray-700 px-8 py-4 rounded-lg text-lg font-semibold border border-gray-300 hover:bg-gray-50 transition-colors"
            >
              {isAuthenticated ? 'View Metrics' : 'Sign In'}
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Why Choose Lead HeatScore?
            </h2>
            <p className="text-xl text-gray-600">
              Built for accuracy, designed for simplicity
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="bg-green-50 p-8 rounded-xl">
              <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center mb-6">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Smart Lead Scoring</h3>
              <p className="text-gray-600">
                Instantly identify your hottest prospects. Our AI analyzes behavior patterns to score leads as hot, warm, or cold.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-blue-50 p-8 rounded-xl">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mb-6">
                <Mail className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Personalized Messages</h3>
              <p className="text-gray-600">
                Get ready-to-send emails tailored for each lead. Hot leads get urgency, warm leads get trust-building, cold leads get value.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-yellow-50 p-8 rounded-xl">
              <div className="w-12 h-12 bg-yellow-500 rounded-lg flex items-center justify-center mb-6">
                <Upload className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Instant Setup</h3>
              <p className="text-gray-600">
                Upload your lead list and start getting personalized messages in minutes. No complex setup or training required.
              </p>
            </div>

            {/* Feature 4 */}
            <div className="bg-purple-50 p-8 rounded-xl">
              <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center mb-6">
                <Target className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Higher Conversions</h3>
              <p className="text-gray-600">
                Personalized messages convert better than generic templates. Focus on hot leads first for maximum ROI.
              </p>
            </div>

            {/* Feature 5 */}
            <div className="bg-red-50 p-8 rounded-xl">
              <div className="w-12 h-12 bg-red-500 rounded-lg flex items-center justify-center mb-6">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Save Time</h3>
              <p className="text-gray-600">
                Stop writing emails from scratch. Get professional, personalized messages ready to send in seconds.
              </p>
            </div>

            {/* Feature 6 */}
            <div className="bg-indigo-50 p-8 rounded-xl">
              <div className="w-12 h-12 bg-indigo-500 rounded-lg flex items-center justify-center mb-6">
                <MessageSquare className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Fresh Content</h3>
              <p className="text-gray-600">
                Every message is unique. No repetitive templates - each email is crafted specifically for that lead.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Analytics Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Comprehensive Analytics
            </h2>
            <p className="text-xl text-gray-600">
              Track performance with detailed metrics and insights
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {/* Metric 1 */}
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <BarChart3 className="h-8 w-8 text-blue-600" />
                <span className="text-2xl font-bold text-gray-900">95%</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Accuracy Rate</h3>
              <p className="text-gray-600 text-sm">ML model classification accuracy</p>
            </div>

            {/* Metric 2 */}
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <TrendingUp className="h-8 w-8 text-green-600" />
                <span className="text-2xl font-bold text-gray-900">40%</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Conversion Lift</h3>
              <p className="text-gray-600 text-sm">Improvement with personalized messages</p>
            </div>

            {/* Metric 3 */}
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <Mail className="h-8 w-8 text-purple-600" />
                <span className="text-2xl font-bold text-gray-900">3.2s</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Response Time</h3>
              <p className="text-gray-600 text-sm">Average API response time</p>
            </div>

            {/* Metric 4 */}
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <MessageSquare className="h-8 w-8 text-orange-600" />
                <span className="text-2xl font-bold text-gray-900">24/7</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Uptime</h3>
              <p className="text-gray-600 text-sm">System availability guarantee</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-blue-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold text-white mb-6">
            Ready to Transform Your Lead Management?
          </h2>
          <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
            Join thousands of businesses using AI-powered lead scoring to increase conversions and optimize their sales process.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={handleGetStarted}
              className="bg-white text-blue-600 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-gray-50 transition-colors flex items-center justify-center"
            >
              {isAuthenticated ? 'Go to Dashboard' : 'Get Started'}
              <ArrowRight className="ml-2 h-5 w-5" />
            </button>
            <button 
              onClick={handleViewDemo}
              className="bg-blue-700 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-800 transition-colors"
            >
              {isAuthenticated ? 'View Metrics' : 'View Demo'}
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Target className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">Lead HeatScore</span>
            </div>
            <div className="text-gray-400 text-sm">
              © 2024 Lead HeatScore. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
