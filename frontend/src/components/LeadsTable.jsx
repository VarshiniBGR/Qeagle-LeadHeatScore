import React, { useState, useMemo } from 'react';
import { 
  ChevronDown, 
  ChevronUp, 
  Search, 
  Filter,
  TrendingUp,
  TrendingDown,
  Minus,
  Eye,
  MessageCircle
} from 'lucide-react';
import ProbabilityBar from './ProbabilityBar';
import RecommendationPanel from './RecommendationPanel';

const LeadsTable = ({ leads, loading = false }) => {
  const [sortField, setSortField] = useState('confidence');
  const [sortDirection, setSortDirection] = useState('desc');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterHeatScore, setFilterHeatScore] = useState('all');
  const [selectedLead, setSelectedLead] = useState(null);
  const [viewMode, setViewMode] = useState('view'); // 'view' or 'message'
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  const sortedAndFilteredLeads = useMemo(() => {
    let filtered = leads.filter(lead => {
      const matchesSearch = 
        (lead.lead_data.name && lead.lead_data.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (lead.lead_data.email && lead.lead_data.email.toLowerCase().includes(searchTerm.toLowerCase())) ||
        lead.lead_data.source.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lead.lead_data.role.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lead.lead_data.region.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lead.lead_data.campaign.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (lead.lead_data.search_keywords && lead.lead_data.search_keywords.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesHeatScore = filterHeatScore === 'all' || lead.score.heat_score === filterHeatScore;
      
      return matchesSearch && matchesHeatScore;
    });

    // Keep original order - no sorting needed since map preserves order
    return filtered;
  }, [leads, searchTerm, filterHeatScore]);

  // Pagination logic
  const totalPages = Math.ceil(sortedAndFilteredLeads.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedLeads = sortedAndFilteredLeads.slice(startIndex, endIndex);

  // Reset to first page when filters change
  React.useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, filterHeatScore, sortField, sortDirection]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const getHeatScoreIcon = (heatScore) => {
    switch (heatScore) {
      case 'hot':
        return <TrendingUp className="h-4 w-4 text-danger-500" />;
      case 'warm':
        return <Minus className="h-4 w-4 text-warning-500" />;
      case 'cold':
        return <TrendingDown className="h-4 w-4 text-gray-400" />;
      default:
        return null;
    }
  };

  const getHeatScoreColor = (heatScore) => {
    switch (heatScore) {
      case 'hot':
        return 'bg-red-100 text-red-800';
      case 'warm':
        return 'bg-yellow-100 text-yellow-800';
      case 'cold':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getChannelIcon = (channel) => {
    switch (channel) {
      case 'email':
        return '📧';
      case 'sms':
        return '💬'; // WhatsApp icon for SMS
      case 'linkedin':
        return '💼';
      case 'whatsapp':
        return '💬';
      case 'social':
        return '📱';
      default:
        return '📧';
    }
  };

  const getEmailTypeForLead = (heatScore) => {
    if (heatScore === "hot" || heatScore === "warm") {
      return "rag";
    } else {
      return "template";
    }
  };

  const getEmailTypeColor = (emailType) => {
    switch (emailType) {
      case 'rag':
        return 'bg-green-100 text-green-800';
      case 'template':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getEmailTypeIcon = (emailType) => {
    switch (emailType) {
      case 'rag':
        return '🤖';
      case 'template':
        return '📧';
      default:
        return '❓';
    }
  };

  const getChannelName = (channel) => {
    switch (channel) {
      case 'email':
        return 'Email';
      case 'telegram':
        return 'Telegram';
      case 'newsletter':
        return 'Newsletter';
      default:
        return 'Email';
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          <span className="ml-3 text-gray-600">Loading leads...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search and Filter Controls */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search leads..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input pl-10"
          />
        </div>
        
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-gray-400" />
          <select
            value={filterHeatScore}
            onChange={(e) => setFilterHeatScore(e.target.value)}
            className="input w-auto"
          >
            <option value="all">All Heat Scores</option>
            <option value="hot">Hot</option>
            <option value="warm">Warm</option>
            <option value="cold">Cold</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Lead Info
                </th>
                <th 
                  className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('heat_score')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Heat</span>
                    {sortField === 'heat_score' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
                    )}
                  </div>
                </th>
                <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Email Type
                </th>
                <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Channel Strategy
                </th>
                <th 
                  className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('confidence')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Conf</span>
                    {sortField === 'confidence' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
                    )}
                  </div>
                </th>
                <th 
                  className="hidden md:table-cell px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('recency_days')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Days</span>
                    {sortField === 'recency_days' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
                    )}
                  </div>
                </th>
                <th 
                  className="hidden lg:table-cell px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('page_views')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Views</span>
                    {sortField === 'page_views' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
                    )}
                  </div>
                </th>
                <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Channel
                </th>
                <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {paginatedLeads.map((lead, index) => (
                <tr key={lead.lead_id || lead.lead_data?.lead_id || index} className="hover:bg-gray-50">
                  <td className="px-3 py-3">
                    <div className="text-sm font-medium text-gray-900 truncate">
                      {lead.lead_data.name || 'Unknown Name'}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                      {lead.lead_data.email || 'No email'}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                      📱 {lead.lead_data.phone || 'No phone'}
                    </div>
                    <div className="text-xs text-gray-400 truncate">
                      {lead.lead_data.role} • {lead.lead_data.region}
                    </div>
                  </td>
                  
                  
                  <td className="px-2 py-3">
                    <div className="flex items-center space-x-1">
                      {getHeatScoreIcon(lead.score.heat_score)}
                      <span className={`inline-flex px-1.5 py-0.5 text-xs font-semibold rounded-full ${getHeatScoreColor(lead.score.heat_score)}`}>
                        {lead.score.heat_score.toUpperCase()}
                      </span>
                    </div>
                  </td>
                  
                <td className="px-2 py-3">
                  <div className="flex items-center space-x-1">
                    <span className="text-sm">{getEmailTypeIcon(getEmailTypeForLead(lead.score.heat_score))}</span>
                    <span className={`inline-flex px-1.5 py-0.5 text-xs font-semibold rounded-full ${getEmailTypeColor(getEmailTypeForLead(lead.score.heat_score))}`}>
                      {getEmailTypeForLead(lead.score.heat_score).toUpperCase()}
                    </span>
                  </div>
                </td>
                <td className="px-2 py-3">
                  <div className="flex items-center space-x-1">
                    <span className="text-sm">
                      {lead.score.heat_score === "hot" && "📱"}
                      {lead.score.heat_score === "warm" && "📧"}
                      {lead.score.heat_score === "cold" && "📰"}
                    </span>
                    <span className={`inline-flex px-1.5 py-0.5 text-xs font-semibold rounded-full ${
                      lead.score.heat_score === "hot" ? "bg-red-100 text-red-800" :
                      lead.score.heat_score === "warm" ? "bg-blue-100 text-blue-800" :
                      "bg-gray-100 text-gray-800"
                    }`}>
                      {lead.score.heat_score === "hot" && "Telegram"}
                      {lead.score.heat_score === "warm" && "RAG Email"}
                      {lead.score.heat_score === "cold" && "Newsletter"}
                    </span>
                  </div>
                </td>
                  
                  <td className="px-2 py-3">
                    <div className="text-xs text-gray-900">
                      {(lead.score.confidence * 100).toFixed(0)}%
                    </div>
                  </td>
                  
                  <td className="hidden md:table-cell px-2 py-3 text-xs text-gray-900">
                    {lead.lead_data.recency_days}d
                  </td>
                  
                  <td className="hidden lg:table-cell px-2 py-3 text-xs text-gray-900">
                    {lead.lead_data.page_views}
                  </td>
                  
                  <td className="px-2 py-3">
                    {lead.recommendation && (
                      <div className="flex items-center space-x-1">
                        <span className="text-sm">{getChannelIcon(lead.recommendation.recommended_channel)}</span>
                        <span className="text-xs font-medium text-gray-900 truncate">
                          {getChannelName(lead.recommendation.recommended_channel)}
                        </span>
                      </div>
                    )}
                  </td>
                  
                  <td className="px-2 py-3 text-sm font-medium">
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => {
                          setViewMode('view');
                          setSelectedLead(lead);
                        }}
                        className="inline-flex items-center px-2 py-1 border border-gray-300 shadow-sm text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-1 focus:ring-offset-1 focus:ring-primary-500 transition-colors duration-200"
                        title="View Details"
                      >
                        <Eye className="h-3 w-3" />
                      </button>
                      {lead.recommendation && (
                        <button
                          onClick={() => {
                            setViewMode('message');
                            setSelectedLead(lead);
                          }}
                          className="inline-flex items-center px-2 py-1 border border-transparent text-xs font-medium rounded text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-1 focus:ring-offset-1 focus:ring-primary-500 transition-colors duration-200"
                          title={`Send ${lead.score.heat_score === "hot" ? "Telegram Message" : lead.score.heat_score === "warm" ? "RAG Email" : "Newsletter"}`}
                        >
                          <MessageCircle className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {sortedAndFilteredLeads.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-500">No leads found matching your criteria</div>
          </div>
        )}

        {/* Pagination Controls */}
        {sortedAndFilteredLeads.length > 0 && (
          <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
            <div className="flex-1 flex justify-between sm:hidden">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Showing <span className="font-medium">{startIndex + 1}</span> to{' '}
                  <span className="font-medium">{Math.min(endIndex, sortedAndFilteredLeads.length)}</span> of{' '}
                  <span className="font-medium">{sortedAndFilteredLeads.length}</span> results
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <div className="flex items-center">
                  <label htmlFor="items-per-page" className="text-sm text-gray-700 mr-2">
                    Show:
                  </label>
                  <select
                    id="items-per-page"
                    value={itemsPerPage}
                    onChange={(e) => {
                      setItemsPerPage(Number(e.target.value));
                      setCurrentPage(1);
                    }}
                    className="border border-gray-300 rounded-md px-2 py-1 text-sm"
                  >
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                    <option value={20}>20</option>
                    <option value={50}>50</option>
                  </select>
                </div>
                <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span className="sr-only">Previous</span>
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </button>
                  
                  {/* Page numbers */}
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    let pageNum;
                    if (totalPages <= 5) {
                      pageNum = i + 1;
                    } else if (currentPage <= 3) {
                      pageNum = i + 1;
                    } else if (currentPage >= totalPages - 2) {
                      pageNum = totalPages - 4 + i;
                    } else {
                      pageNum = currentPage - 2 + i;
                    }
                    
                    return (
                      <button
                        key={pageNum}
                        onClick={() => setCurrentPage(pageNum)}
                        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                          currentPage === pageNum
                            ? 'z-10 bg-primary-50 border-primary-500 text-primary-600'
                            : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                        }`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}
                  
                  <button
                    onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                    disabled={currentPage === totalPages}
                    className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span className="sr-only">Next</span>
                    <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                    </svg>
                  </button>
                </nav>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Recommendation Panel */}
      {selectedLead && (
        <RecommendationPanel
          lead={selectedLead}
          mode={viewMode}
          onClose={() => setSelectedLead(null)}
        />
      )}
    </div>
  );
};

export default LeadsTable;
