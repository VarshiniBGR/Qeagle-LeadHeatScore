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
  MessageCircle,
  Calendar,
  BarChart3
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
        return <TrendingUp className="h-4 w-4 text-red-500" />;
      case 'warm':
        return <Minus className="h-4 w-4 text-yellow-500" />;
      case 'cold':
        return <TrendingDown className="h-4 w-4 text-gray-400" />;
      default:
        return null;
    }
  };

  const getHeatScoreColor = (heatScore) => {
    switch (heatScore) {
      case 'hot':
        return 'bg-red-50 text-red-700 border-red-200';
      case 'warm':
        return 'bg-yellow-50 text-yellow-700 border-yellow-200';
      case 'cold':
        return 'bg-gray-50 text-gray-700 border-gray-200';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200">
        <div className="flex items-center justify-center py-16">
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin rounded-full h-10 w-10 border-2 border-blue-600 border-t-transparent"></div>
            <span className="text-gray-600 font-medium">Loading leads...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search and Filter Controls */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col lg:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by name, email, role, region, or campaign..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            />
          </div>
          
          <div className="flex items-center space-x-3">
            <Filter className="h-5 w-5 text-gray-400" />
            <select
              value={filterHeatScore}
              onChange={(e) => setFilterHeatScore(e.target.value)}
              className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            >
              <option value="all">All Heat Scores</option>
              <option value="hot">Hot Leads</option>
              <option value="warm">Warm Leads</option>
              <option value="cold">Cold Leads</option>
            </select>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Lead Information
                </th>
                <th 
                  className="px-4 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() => handleSort('heat_score')}
                >
                  <div className="flex items-center space-x-2">
                    <span>Heat Score</span>
                    {sortField === 'heat_score' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() => handleSort('confidence')}
                >
                  <div className="flex items-center space-x-2">
                    <span>Confidence</span>
                    {sortField === 'confidence' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
                    )}
                  </div>
                </th>
                <th 
                  className="hidden md:table-cell px-4 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() => handleSort('recency_days')}
                >
                  <div className="flex items-center space-x-2">
                    <Calendar className="h-4 w-4" />
                    <span>Recency</span>
                    {sortField === 'recency_days' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
                    )}
                  </div>
                </th>
                <th 
                  className="hidden lg:table-cell px-4 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() => handleSort('page_views')}
                >
                  <div className="flex items-center space-x-2">
                    <BarChart3 className="h-4 w-4" />
                    <span>Engagement</span>
                    {sortField === 'page_views' && (
                      sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
                    )}
                  </div>
                </th>
                <th className="px-6 py-4 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {paginatedLeads.map((lead, index) => (
                <tr key={lead.lead_id || lead.lead_data?.lead_id || index} className="hover:bg-gray-50 transition-colors">
                  <td className="px-6 py-4">
                    <div className="flex items-start space-x-3">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold text-gray-900 truncate">
                          {lead.lead_data.name || 'Unknown Name'}
                        </div>
                        <div className="text-xs text-gray-500 truncate">
                          {lead.lead_data.email ? '***@***.***' : 'No email'}
                        </div>
                        <div className="text-xs text-gray-500 truncate">
                          📱 {lead.lead_data.phone ? '***-***-****' : 'No phone'}
                        </div>
                        <div className="text-xs text-gray-400 truncate mt-1">
                          {lead.lead_data.role} • {lead.lead_data.region}
                        </div>
                      </div>
                    </div>
                  </td>
                  
                  <td className="px-4 py-4">
                    <div className="flex items-center space-x-2">
                      {getHeatScoreIcon(lead.score.heat_score)}
                      <span className={`inline-flex px-3 py-1 text-xs font-semibold rounded-full border ${getHeatScoreColor(lead.score.heat_score)}`}>
                        {lead.score.heat_score.toUpperCase()}
                      </span>
                    </div>
                  </td>
                  
                  <td className="px-4 py-4">
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${lead.score.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-900 min-w-[3rem]">
                        {(lead.score.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  
                  <td className="hidden md:table-cell px-4 py-4">
                    <div className="flex items-center space-x-2">
                      <Calendar className="h-4 w-4 text-gray-400" />
                      <span className="text-sm text-gray-900">
                        {lead.lead_data.recency_days}d ago
                      </span>
                    </div>
                  </td>
                  
                  <td className="hidden lg:table-cell px-4 py-4">
                    <div className="flex items-center space-x-2">
                      <BarChart3 className="h-4 w-4 text-gray-400" />
                      <span className="text-sm text-gray-900">
                        {lead.lead_data.page_views} views
                      </span>
                    </div>
                  </td>
                  
                  <td className="px-6 py-4">
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => {
                          setViewMode('view');
                          setSelectedLead(lead);
                        }}
                        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
                        title="View Lead Details (No RAG Loading)"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => {
                          setViewMode('message');
                          setSelectedLead(lead);
                        }}
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
                        title="Generate RAG-Powered Personalized Message"
                      >
                        <MessageCircle className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {sortedAndFilteredLeads.length === 0 && (
          <div className="text-center py-16">
            <div className="text-gray-400 mb-2">
              <Users className="h-12 w-12 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-1">No leads found</h3>
            <p className="text-gray-500">Try adjusting your search criteria or filters</p>
          </div>
        )}

        {/* Pagination Controls */}
        {sortedAndFilteredLeads.length > 0 && (
          <div className="bg-white px-6 py-4 flex items-center justify-between border-t border-gray-200">
            <div className="flex-1 flex justify-between sm:hidden">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Next
              </button>
            </div>
            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Showing <span className="font-semibold">{startIndex + 1}</span> to{' '}
                  <span className="font-semibold">{Math.min(endIndex, sortedAndFilteredLeads.length)}</span> of{' '}
                  <span className="font-semibold">{sortedAndFilteredLeads.length}</span> results
                </p>
              </div>
              <div className="flex items-center space-x-4">
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
                    className="border border-gray-300 rounded-lg px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value={5}>5</option>
                    <option value={10}>10</option>
                    <option value={20}>20</option>
                    <option value={50}>50</option>
                  </select>
                </div>
                <nav className="relative z-0 inline-flex rounded-lg shadow-sm -space-x-px" aria-label="Pagination">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="relative inline-flex items-center px-3 py-2 rounded-l-lg border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <span className="sr-only">Previous</span>
                    <ChevronDown className="h-4 w-4 rotate-90" />
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
                        className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium transition-colors ${
                          currentPage === pageNum
                            ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
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
                    className="relative inline-flex items-center px-3 py-2 rounded-r-lg border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <span className="sr-only">Next</span>
                    <ChevronDown className="h-4 w-4 -rotate-90" />
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