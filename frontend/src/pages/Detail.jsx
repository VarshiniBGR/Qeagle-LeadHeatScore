import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  TrendingUp, 
  TrendingDown, 
  Minus,
  Mail,
  Phone,
  Linkedin,
  Smartphone,
  Share2,
  Copy,
  CheckCircle
} from 'lucide-react';
import ProbabilityBar from '../components/ProbabilityBar';
import RecommendationPanel from '../components/RecommendationPanel';
import { leadAPI } from '../lib/api';
import toast from 'react-hot-toast';

const Detail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [lead, setLead] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadLead();
  }, [id]);

  const loadLead = async () => {
    setLoading(true);
    try {
      // Fetch real lead data from API
      const response = await leadAPI.getLead(id);
      if (response.success) {
        setLead(response.data);
      } else {
        toast.error('Lead not found');
        navigate('/leads');
      }
    } catch (error) {
      console.error('Error loading lead:', error);
      toast.error('Error loading lead details');
      navigate('/leads');
    } finally {
      setLoading(false);
    }
  };

  const getHeatScoreIcon = (heatScore) => {
    switch (heatScore) {
      case 'hot':
        return <TrendingUp className="h-6 w-6 text-danger-500" />;
      case 'warm':
        return <Minus className="h-6 w-6 text-warning-500" />;
      case 'cold':
        return <TrendingDown className="h-6 w-6 text-gray-400" />;
      default:
        return null;
    }
  };

  const getHeatScoreColor = (heatScore) => {
    switch (heatScore) {
      case 'hot':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'warm':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'cold':
        return 'bg-gray-100 text-gray-800 border-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const FeatureImportance = ({ features }) => {
    const sortedFeatures = Object.entries(features)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5);

    return (
      <div className="space-y-3">
        {sortedFeatures.map(([feature, importance]) => (
          <div key={feature} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="font-medium text-gray-700 capitalize">
                {feature.replace('_', ' ')}
              </span>
              <span className="text-gray-500">
                {(importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${importance * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        <span className="ml-3 text-gray-600">Loading lead details...</span>
      </div>
    );
  }

  if (!lead) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Lead Not Found</h2>
        <p className="text-gray-600 mb-6">The requested lead could not be found.</p>
        <button
          onClick={() => navigate('/leads')}
          className="btn btn-primary"
        >
          Back to Leads
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <button
          onClick={() => navigate('/leads')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 mb-4"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Leads</span>
        </button>
        
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Lead Analysis: {lead.lead_id}
            </h1>
            <p className="text-gray-600">
              Detailed AI-powered analysis and recommendations
            </p>
          </div>
          
          <div className={`flex items-center space-x-3 px-4 py-2 rounded-lg border ${getHeatScoreColor(lead.score.heat_score)}`}>
            {getHeatScoreIcon(lead.score.heat_score)}
            <span className="font-semibold text-lg">
              {lead.score.heat_score.toUpperCase()}
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Lead Information */}
        <div className="lg:col-span-2 space-y-6">
          {/* Basic Info */}
          <div className="card">
            <h3 className="font-medium text-gray-900 mb-4">Lead Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Source</label>
                <p className="text-gray-900 capitalize">{lead.lead_data.source}</p>
              </div>
              <div>
                <label className="label">Region</label>
                <p className="text-gray-900">{lead.lead_data.region}</p>
              </div>
              <div>
                <label className="label">Role</label>
                <p className="text-gray-900 capitalize">{lead.lead_data.role}</p>
              </div>
              <div>
                <label className="label">Campaign</label>
                <p className="text-gray-900 capitalize">{lead.lead_data.campaign}</p>
              </div>
              <div>
                <label className="label">Recency</label>
                <p className="text-gray-900">{lead.lead_data.recency_days} days ago</p>
              </div>
              <div>
                <label className="label">Page Views</label>
                <p className="text-gray-900">{lead.lead_data.page_views}</p>
              </div>
              <div>
                <label className="label">Last Touch</label>
                <p className="text-gray-900 capitalize">{lead.lead_data.last_touch}</p>
              </div>
              <div>
                <label className="label">Prior Interest</label>
                <p className="text-gray-900 capitalize">{lead.lead_data.prior_course_interest}</p>
              </div>
            </div>
          </div>

          {/* Classification Details */}
          <div className="card">
            <h3 className="font-medium text-gray-900 mb-4">Classification Analysis</h3>
            
            <div className="space-y-6">
              <div>
                <label className="label">Probability Distribution</label>
                <ProbabilityBar 
                  probabilities={lead.score.probabilities}
                  highlight={lead.score.heat_score}
                />
              </div>
              
              <div>
                <label className="label">Confidence Score</label>
                <div className="flex items-center space-x-3">
                  <div className="flex-1 bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-primary-600 h-3 rounded-full transition-all duration-300"
                      style={{ width: `${lead.score.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-gray-700">
                    {(lead.score.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              
              <div>
                <label className="label">Feature Importance</label>
                <FeatureImportance features={lead.score.features_importance} />
              </div>
            </div>
          </div>
        </div>

        {/* Recommendations */}
        <div className="space-y-6">
          <RecommendationPanel recommendation={lead.recommendation} />
        </div>
      </div>
    </div>
  );
};

export default Detail;
