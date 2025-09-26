import React, { useState, useEffect } from 'react';
import { 
  MessageCircle, 
  Send, 
  Edit3, 
  X,
  Mail,
  MessageSquare,
  Linkedin,
  Smartphone,
  Eye,
  EyeOff,
  Clock,
  MousePointer,
  Search,
  FileText
} from 'lucide-react';
import toast from 'react-hot-toast';
import { leadAPI } from '../lib/api';

const RecommendationPanel = ({ lead, mode = 'view', onClose }) => {
  const [personalizedEmail, setPersonalizedEmail] = useState(null);
  const [loadingEmail, setLoadingEmail] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedMessage, setEditedMessage] = useState(lead.recommendation?.message_content || '');
  const [isSending, setIsSending] = useState(false);
  const [emailType, setEmailType] = useState("template"); // "template" or "rag"
  const [smartStrategy, setSmartStrategy] = useState(null);

  // Determine smart email type based on lead heat score
  const getSmartEmailType = (heatScore) => {
    if (heatScore === "hot" || heatScore === "warm") {
      return "rag";
    } else {
      return "template";
    }
  };

  // Set initial email type based on smart strategy
  useEffect(() => {
    if (lead?.heat_score) {
      const smartType = getSmartEmailType(lead.heat_score);
      setEmailType(smartType);
      setSmartStrategy(`${lead.heat_score} leads get ${smartType === 'rag' ? 'Smart' : 'Template'} emails`);
    }
  }, [lead?.heat_score]);

  // Load personalized email when in message mode
  useEffect(() => {
    const loadPersonalizedEmail = async () => {
      if (mode !== 'message' || !lead?.lead_data) return;
      
      setLoadingEmail(true);
      try {
        // For all channels, use the API to get fresh personalized content
        const emailData = await leadAPI.getPersonalizedEmail(lead.lead_data);
        console.log('Received personalized data:', emailData);
        
        // For Telegram messages, convert markdown formatting to HTML
        if (lead.recommendation?.recommended_channel === 'telegram') {
          let htmlContent = emailData.text_content || emailData.html_content || ''
            .replace(/\n/g, '<br>')
            .replace(/~~(.*?)~~/g, '<del>$1</del>') // Convert strikethrough
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Convert bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>'); // Convert italic
          
          const telegramData = {
            ...emailData,
            html_content: htmlContent,
            email_type: 'telegram',
            final_email_type: 'telegram',
            smart_strategy: `${lead.score?.heat_score || 'hot'} leads get Telegram messages`,
            personalization_data: {
              name: lead.lead_data.name,
              role: lead.lead_data.role,
              search_keywords: lead.lead_data.search_keywords || '',
              page_views: lead.lead_data.page_views || 0,
              time_spent: lead.lead_data.time_spent || 0,
              course_actions: lead.lead_data.course_actions || '',
              prior_course_interest: lead.lead_data.prior_course_interest || 'low'
            }
          };
          setPersonalizedEmail(telegramData);
        } else {
          setPersonalizedEmail(emailData);
        }
      } catch (error) {
        console.error('Error loading personalized content:', error);
        toast.error('Failed to load personalized content');
      } finally {
        setLoadingEmail(false);
      }
    };

    loadPersonalizedEmail();
  }, [lead, mode]);

  if (!lead.recommendation) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
          <div className="text-center">
            <MessageCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Recommendation Available</h3>
            <p className="text-gray-500 mb-4">This lead doesn't have a recommendation yet.</p>
            <button
              onClick={onClose}
              className="btn btn-secondary"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  const getChannelIcon = (channel) => {
    switch (channel) {
      case 'email':
        return <Mail className="h-5 w-5" />;
      case 'telegram':
        return <MessageCircle className="h-5 w-5" />;
      case 'whatsapp':
        return <Smartphone className="h-5 w-5" />;
      case 'newsletter':
        return <FileText className="h-5 w-5" />;
      default:
        return <Mail className="h-5 w-5" />;
    }
  };

  const getChannelName = (channel) => {
    switch (channel) {
      case 'email':
        return 'Email';
      case 'telegram':
        return 'Telegram';
      case 'whatsapp':
        return 'WhatsApp';
      case 'newsletter':
        return 'Newsletter';
      default:
        return 'Email';
    }
  };

  const getChannelColor = (channel) => {
    switch (channel) {
      case 'email':
        return 'bg-blue-100 text-blue-800';
      case 'telegram':
        return 'bg-red-100 text-red-800';
      case 'whatsapp':
        return 'bg-green-100 text-green-800';
      case 'newsletter':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-blue-100 text-blue-800';
    }
  };

  const handleSendMessage = async () => {
    const message = isEditing ? editedMessage : lead.recommendation.message_content;
    const channel = lead.recommendation.recommended_channel;
    
    // Check required fields based on channel
    if ((channel === 'telegram' || channel === 'whatsapp') && !lead.lead_data?.phone) {
      toast.error('No phone number found for messaging');
      return;
    } else if (channel === 'email' && !lead.lead_data?.email) {
      toast.error('No email address found for this lead');
      return;
    }

    setIsSending(true);
    
    try {
      let result;
      
      if (channel === 'telegram') {
        // Send Telegram message
        result = await leadAPI.sendTelegramMessage(
          lead.lead_data.phone,
          message,
          lead.lead_data
        );
      } else if (channel === 'whatsapp') {
        // Send WhatsApp message
        result = await leadAPI.sendWhatsAppMessage(
          lead.lead_data.phone,
          message
        );
      } else {
        // Send email (default behavior)
        // Prepare lead data for email sending - send complete lead data
        const leadData = {
          name: lead.lead_data.name || 'Valued Customer',
          email: lead.lead_data.email,
          source: lead.lead_data.source || 'unknown',
          recency_days: lead.lead_data.recency_days || 0,
          region: lead.lead_data.region || 'unknown',
          role: lead.lead_data.role || 'professional',
          campaign: lead.lead_data.campaign || 'Lead HeatScore Campaign',
          page_views: lead.lead_data.page_views || 0,
          last_touch: lead.lead_data.last_touch || 'unknown',
          prior_course_interest: lead.lead_data.prior_course_interest || 'low'
        };

        // Send email via API
        result = await leadAPI.sendEmail(
          lead.lead_id,
          lead.lead_data.email,
          leadData,
          emailType  // Pass email type
        );
      }

      // Show success message based on channel
      if (channel === 'telegram') {
        toast.success(`Telegram message sent successfully to ${lead.lead_data.phone}!`);
      } else if (channel === 'newsletter') {
        toast.success(`Newsletter sent successfully to ${lead.lead_data.email}!`);
      } else {
        toast.success(`RAG Email sent successfully to ${lead.lead_data.email}!`);
      }
      
      console.log('Message sent:', result);
      
      // Close the panel after successful send
      setTimeout(() => {
        onClose();
      }, 1500);
      
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error(`Failed to send ${channel === 'telegram' ? 'Telegram message' : channel === 'newsletter' ? 'Newsletter' : 'RAG Email'}. Please check your configuration.`);
    } finally {
      setIsSending(false);
    }
  };



  const handleRegenerate = () => {
    toast.success('Regenerating recommendation...');
    // In a real app, you would call the API to regenerate
    console.log('Regenerating recommendation for lead:', lead.lead_id);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MessageCircle className="h-6 w-6 text-primary-600" />
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  {mode === 'view' ? 'Lead Engagement Details' : 
                   lead.recommendation?.recommended_channel === 'telegram' ? 'Telegram Message Preview' :
                   lead.recommendation?.recommended_channel === 'newsletter' ? 'Newsletter Preview' :
                   'Personalized Email Preview'}
                </h2>
                <p className="text-sm text-gray-500">
                  {mode === 'view' 
                    ? 'View detailed engagement metrics and behavioral data' 
                    : lead.recommendation?.recommended_channel === 'telegram' ? 'AI-powered Telegram message with personalized content and call-to-action' :
                      lead.recommendation?.recommended_channel === 'newsletter' ? 'Newsletter content for low-touch nurturing and engagement' :
                      'Coursera-style personalized email with search keywords and course recommendations'
                  }
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-6 w-6" />
            </button>
          </div>
          
          {/* Channel and Status Tags */}
          <div className="flex items-center space-x-2 mt-4">
            <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getChannelColor(lead.recommendation.recommended_channel)}`}>
              {getChannelIcon(lead.recommendation.recommended_channel)}
              <span className="ml-2">{getChannelName(lead.recommendation.recommended_channel)}</span>
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-800">
              Recommended
            </span>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* View Mode: Show detailed lead data */}
          {mode === 'view' && (
            <div>
              <div className="flex items-center space-x-2 mb-3">
                <Eye className="h-5 w-5 text-gray-600" />
                <h3 className="font-medium text-gray-900">Lead Engagement Details:</h3>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <MousePointer className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-900">Page Views</span>
                  </div>
                  <div className="text-2xl font-bold text-blue-600">
                    {lead.lead_data.page_views || 0}
                  </div>
                </div>
                
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Clock className="h-4 w-4 text-green-600" />
                    <span className="text-sm font-medium text-green-900">Time Spent</span>
                  </div>
                  <div className="text-2xl font-bold text-green-600">
                    {lead.lead_data.time_spent ? Math.round(lead.lead_data.time_spent / 60) : 0}m
                  </div>
                </div>
                
                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Search className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium text-purple-900">Search Keywords</span>
                  </div>
                  <div className="text-sm font-medium text-purple-600 truncate">
                    {lead.lead_data.search_keywords || 'N/A'}
                  </div>
                </div>
                
                <div className="bg-orange-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <MessageCircle className="h-4 w-4 text-orange-600" />
                    <span className="text-sm font-medium text-orange-900">Course Actions</span>
                  </div>
                  <div className="text-sm font-medium text-orange-600">
                    {lead.lead_data.course_actions || 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Message Mode: Show personalized email */}
          {mode === 'message' && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <Mail className="h-5 w-5 text-gray-600" />
                  <h3 className="font-medium text-gray-900">Personalized Email:</h3>
                </div>
              </div>
              
              {mode === 'message' ? (
                loadingEmail ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
                    <span className="ml-3 text-gray-600">
                      {lead.recommendation?.recommended_channel === 'telegram' ? 'Generating Telegram message...' :
                       lead.recommendation?.recommended_channel === 'newsletter' ? 'Generating newsletter content...' :
                       'Generating personalized email...'}
                    </span>
                  </div>
                ) : personalizedEmail ? (
                  <div className="space-y-4">
                    {/* Email Subject */}
                    <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-400">
                      <div className="text-sm font-medium text-blue-900 mb-1">Subject Line:</div>
                      <div className="text-lg font-semibold text-blue-800">
                        {personalizedEmail.subject}
                      </div>
                    </div>
                    
                    {/* Channel-Specific Message Preview */}
                    <div className="bg-gray-50 rounded-lg p-4 border">
                      <div className="text-sm font-medium text-gray-700 mb-2">
                        {lead.recommendation.recommended_channel === 'telegram' && '📱 Telegram Message Preview:'}
                        {lead.recommendation.recommended_channel === 'email' && '📧 Email Preview:'}
                        {lead.recommendation.recommended_channel === 'newsletter' && '📰 Newsletter Preview:'}
                        {!['telegram', 'email', 'newsletter'].includes(lead.recommendation.recommended_channel) && 'Message Preview:'}
                      </div>
                      
                      {/* Smart Strategy Info */}
                      {personalizedEmail.smart_strategy && (
                        <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                          <div className="flex items-center">
                            <div className="flex-shrink-0">
                              <svg className="h-4 w-4 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                              </svg>
                            </div>
                            <div className="ml-2">
                              <p className="text-xs font-medium text-blue-800">
                                Smart Strategy: {personalizedEmail.smart_strategy}
                              </p>
                              <p className="text-xs text-blue-700">
                                Email Type: {personalizedEmail.email_type?.toUpperCase() || 'TEMPLATE'}
                                {personalizedEmail.email_type === 'rag' && (
                                  <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gradient-to-r from-pink-500 to-cyan-500 text-white">
                                    ✨ AI-POWERED
                                  </span>
                                )}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      <div className="bg-white rounded-lg border overflow-hidden shadow-lg">
                        <style>
                          {`
                            .telegram-preview del {
                              text-decoration: line-through;
                              color: #ef4444 !important;
                              opacity: 0.8;
                            }
                            .telegram-preview strong {
                              color: #fbbf24 !important;
                              font-weight: bold;
                            }
                          `}
                        </style>
                        {personalizedEmail.html_content ? (
                          <div 
                            className={`email-preview max-h-96 overflow-y-auto ${
                              lead.recommendation.recommended_channel === 'telegram' ? 'bg-gray-900 text-green-400 p-4 font-mono text-sm whitespace-pre-wrap telegram-preview' :
                              lead.recommendation.recommended_channel === 'newsletter' ? 'bg-gray-50 p-4' :
                              'p-4'
                            }`}
                            style={{
                              ...(lead.recommendation.recommended_channel === 'telegram' && {
                                '--tw-text-opacity': '1',
                                color: 'rgb(34 197 94 / var(--tw-text-opacity))'
                              })
                            }}
                            dangerouslySetInnerHTML={{ __html: personalizedEmail.html_content }}
                          />
                        ) : (
                          <div className="p-4 text-gray-500 text-center">
                            No content available
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Personalization Data */}
                    <div className="bg-green-50 rounded-lg p-4">
                      <div className="text-sm font-medium text-green-900 mb-2">Personalization Applied:</div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div><strong>Name:</strong> {personalizedEmail.personalization_data.name}</div>
                        <div><strong>Role:</strong> {personalizedEmail.personalization_data.role}</div>
                        <div><strong>Keywords:</strong> {personalizedEmail.personalization_data.search_keywords}</div>
                        <div><strong>Interest:</strong> {personalizedEmail.personalization_data.prior_course_interest}</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-50 rounded-lg p-4 border">
                    <p className="text-gray-800 leading-relaxed">
                      {lead.recommendation.message_content}
                    </p>
                  </div>
                )
              ) : (
                <div className="bg-gray-50 rounded-lg p-4 border">
                  {isEditing ? (
                    <textarea
                      value={editedMessage}
                      onChange={(e) => setEditedMessage(e.target.value)}
                      className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      placeholder="Edit your message..."
                    />
                  ) : (
                    <p className="text-gray-800 leading-relaxed">
                      {lead.recommendation.message_content}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Rationale */}
          <div>
            <h3 className="font-medium text-gray-900 mb-3">Rationale</h3>
            <p className="text-gray-600 text-sm leading-relaxed">
              {lead.recommendation.rationale}
            </p>
          </div>

          {/* References */}
          <div>
            <h3 className="font-medium text-gray-900 mb-3">References</h3>
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800">
                policy_doc_engagement
              </span>
              <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800">
                persona_{lead.lead_data.role.toLowerCase().replace(' ', '_')}
              </span>
            </div>
          </div>
        </div>

        {/* Actions */}
        {mode === 'message' && (
          <div className="p-6 border-t border-gray-200 bg-gray-50">
            {/* Email Type Selector */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Type
              </label>
              
              {/* Smart Strategy Indicator */}
              {smartStrategy && (
                <div className="mb-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-blue-800">
                        Smart Strategy
                      </p>
                      <p className="text-sm text-blue-700">
                        {smartStrategy}
                      </p>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="flex space-x-4">
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="emailType"
                    value="template"
                    checked={emailType === "template"}
                    onChange={(e) => setEmailType(e.target.value)}
                    className="mr-2 text-blue-600"
                  />
                  <span className="text-sm text-gray-700">Static Template</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="emailType"
                    value="rag"
                    checked={emailType === "rag"}
                    onChange={(e) => setEmailType(e.target.value)}
                    className="mr-2 text-blue-600"
                  />
                  <span className="text-sm text-gray-700">AI Personalized (Smart)</span>
                </label>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {emailType === "template" 
                  ? "Uses predefined template with variable substitution"
                  : "AI-powered personalized content tailored to each lead's interests and behavior"
                }
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={handleSendMessage}
                disabled={isSending}
                className="btn btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {getChannelIcon(lead.recommendation.recommended_channel)}
                <span>
                  {isSending ? 'Sending...' : 
                    lead.recommendation.recommended_channel === 'telegram' ? 'Send Telegram Message' :
                    lead.recommendation.recommended_channel === 'whatsapp' ? 'Send WhatsApp Message' :
                    lead.recommendation.recommended_channel === 'newsletter' ? 'Send Newsletter' :
                    `Send ${emailType === 'rag' ? 'Smart' : 'Template'} Email`
                  }
                </span>
              </button>
              
              <button
                onClick={onClose}
                className="btn btn-secondary flex items-center justify-center space-x-2"
              >
                <X className="h-4 w-4" />
                <span>Close</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RecommendationPanel;