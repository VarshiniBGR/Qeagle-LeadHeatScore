import React, { useState, useEffect } from 'react';
import { Users, TrendingUp, TrendingDown, Minus, Download, RefreshCw, Upload, AlertCircle } from 'lucide-react';
import LeadsTable from '../components/LeadsTable';
import { leadAPI } from '../lib/api';
import toast from 'react-hot-toast';

const Leads = () => {
  const [leads, setLeads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0,
    hot: 0,
    warm: 0,
    cold: 0
  });

  useEffect(() => {
    loadLeads();
  }, []);

  const loadLeads = async () => {
    setLoading(true);
    try {
      // Fetch real leads from the API
      const apiLeads = await leadAPI.getLeads();
      console.log('API returned leads:', apiLeads.length);
      
      if (apiLeads.length > 0) {
        // Use real leads from API
        setLeads(apiLeads);
        
        // Calculate stats
        const newStats = {
          total: apiLeads.length,
          hot: apiLeads.filter(lead => lead.score.heat_score === 'hot').length,
          warm: apiLeads.filter(lead => lead.score.heat_score === 'warm').length,
          cold: apiLeads.filter(lead => lead.score.heat_score === 'cold').length
        };
        setStats(newStats);
        console.log('Using real leads from API:', newStats);
      } else {
        // No leads from API - show empty state
        setLeads([]);
        setStats({ total: 0, hot: 0, warm: 0, cold: 0 });
        console.log('No leads found in API');
      }
      
    } catch (error) {
      console.error('Error loading leads from API:', error);
      // Show empty state instead of fallback sample data
      setLeads([]);
      setStats({ total: 0, hot: 0, warm: 0, cold: 0 });
      toast.error('Error loading leads from server');
    } finally {
      setLoading(false);
    }
  };


  const exportLeads = () => {
    // Convert leads to CSV format
    const csvContent = [
      ['Lead ID', 'Heat Score', 'Confidence', 'Source', 'Region', 'Role', 'Campaign', 'Recency Days', 'Page Views', 'Recommended Channel'],
      ...leads.map(lead => [
        lead.lead_id,
        lead.score.heat_score,
        lead.score.confidence.toFixed(3),
        lead.lead_data.source,
        lead.lead_data.region,
        lead.lead_data.role,
        lead.lead_data.campaign,
        lead.lead_data.recency_days,
        lead.lead_data.page_views,
        lead.recommendation?.recommended_channel || 'N/A'
      ])
    ].map(row => row.join(',')).join('\n');

    // Download CSV
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `leads_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
    
    toast.success('Leads exported successfully');
  };

  const StatCard = ({ title, value, icon: Icon, color, trend }) => (
    <div className="card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
        <div className={`p-3 rounded-full ${color}`}>
          <Icon className="h-6 w-6 text-white" />
        </div>
      </div>
      {trend && (
        <div className="mt-2">
          <span className={`text-sm ${trend > 0 ? 'text-success-600' : 'text-danger-600'}`}>
            {trend > 0 ? '+' : ''}{trend}% from last week
          </span>
        </div>
      )}
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Lead Results</h1>
            <p className="text-gray-600">
              AI-powered lead classification and next action recommendations
            </p>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={loadLeads}
              className="btn btn-secondary"
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
            
            <button
              onClick={exportLeads}
              className="btn btn-primary"
              disabled={leads.length === 0}
            >
              <Download className="h-4 w-4 mr-2" />
              Export CSV
            </button>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Leads"
          value={stats.total}
          icon={Users}
          color="bg-primary-500"
        />
        <StatCard
          title="Hot Leads"
          value={stats.hot}
          icon={TrendingUp}
          color="bg-danger-500"
        />
        <StatCard
          title="Warm Leads"
          value={stats.warm}
          icon={Minus}
          color="bg-warning-500"
        />
        <StatCard
          title="Cold Leads"
          value={stats.cold}
          icon={TrendingDown}
          color="bg-gray-500"
        />
      </div>

      {/* Leads Table */}
      {leads.length === 0 && !loading ? (
        <div className="text-center py-12">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-8 max-w-2xl mx-auto">
            <Upload className="h-16 w-16 text-blue-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-4">No Leads Found</h2>
            <p className="text-gray-600 mb-6">
              You need to upload your CSV data first to see leads and their classifications.
            </p>
            <div className="space-y-4">
              <div className="bg-white p-4 rounded-lg border border-blue-200">
                <h3 className="font-semibold text-gray-900 mb-2">To get started:</h3>
                <ol className="text-left text-sm text-gray-600 space-y-1">
                  <li>1. Go to the <strong>Upload</strong> page</li>
                  <li>2. Upload your CSV file with lead data</li>
                  <li>3. The system will process and classify your leads</li>
                  <li>4. Return here to view your classified leads</li>
                </ol>
              </div>
              <button
                onClick={() => window.location.href = '/upload'}
                className="btn btn-primary"
              >
                <Upload className="h-4 w-4 mr-2" />
                Go to Upload Page
              </button>
            </div>
          </div>
        </div>
      ) : (
        <LeadsTable leads={leads} loading={loading} />
      )}
    </div>
  );
};

export default Leads;
