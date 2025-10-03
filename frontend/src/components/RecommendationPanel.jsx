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

// Minimal personalized fallback content
const getMinimalFallbackSubject = (leadData) => {
  return `Hi ${leadData.name} - Let's Connect`;
};

const getMinimalFallbackContent = (leadData) => {
  const name = leadData.name || 'there';
  const role = leadData.role || 'professional';
  const campaign = leadData.campaign || 'our program';
  
  return `Hi ${name},

I noticed your interest in our ${campaign} program. As a ${role}, I believe this could be valuable for your career.

I'd love to discuss how we can help you achieve your goals.

Best regards,
LearnSprout Team`;
};

// Function to clean rationale from subject lines
const cleanSubjectLine = (subject, heatScore, leadData) => {
  if (!subject) return getMinimalFallbackSubject(leadData);
  
  // Check if subject contains rationale words
  const rationaleWords = ['leverages', 'emphasizing', 'creating', 'acknowledges', 'making', 'because', 'this email', 'by emphasizing', 'while also', 'personal touch'];
  
  const isRationale = rationaleWords.some(word => subject.toLowerCase().includes(word));
  
  if (isRationale) {
    // Return a proper subject line instead of rationale
    return getMinimalFallbackSubject(leadData);
  }
  
  return subject;
};

const RecommendationPanel = ({ lead, mode = 'view', onClose }) => {
  const [personalizedEmail, setPersonalizedEmail] = useState(null);
  const [loadingEmail, setLoadingEmail] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editedMessage, setEditedMessage] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [emailType, setEmailType] = useState("rag"); // All leads use RAG personalization
  const [smartStrategy, setSmartStrategy] = useState(null);

  // Determine smart email type based on lead heat score
  const getSmartEmailType = (heatScore) => {
    // All leads use RAG personalization with different tones based on heat score
    return "rag";  // All leads get RAG-personalized emails
  };

  // Set initial email type based on smart strategy
  useEffect(() => {
    if (lead?.heat_score) {
      const smartType = getSmartEmailType(lead.heat_score);
      setEmailType(smartType);
      // No need for strategy explanation since all leads get RAG emails
      setSmartStrategy(null);
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
        
        // All leads use RAG-personalized emails with different tones
        const ragData = {
          ...emailData,
          email_type: 'rag',
          final_email_type: 'rag',
          // No strategy needed since all leads get RAG emails
          smart_strategy: null,
          // Map API response fields to expected UI fields
          content: emailData.content || emailData.message_content || emailData.body || getMinimalFallbackContent(lead.lead_data),
          subject: cleanSubjectLine(emailData.subject || emailData.subject_line, lead.score?.heat_score || 'hot', lead.lead_data),
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
        
        setPersonalizedEmail(ragData);
      } catch (error) {
        console.error('Error loading personalized content:', error);
        toast.error('Failed to load personalized content');
      } finally {
        setLoadingEmail(false);
      }
    };

    loadPersonalizedEmail();
  }, [lead, mode]);

  // Create a fallback recommendation if none exists
  const fallbackRecommendation = {
    lead_id: lead.lead_id,
    recommended_channel: "email",
    message_content: `Hi ${lead.lead_data?.name || 'there'}! 

Thank you for your interest in our AI program. Based on your profile as a ${lead.lead_data?.role || 'professional'}, we have personalized content ready for you.

Would you like to see our tailored recommendations?`,
    rationale: "Generated fallback recommendation",
    citations: [],
    confidence: 0.8
  };

  const recommendation = lead.recommendation || fallbackRecommendation;
  
  // Initialize edited message when recommendation changes
  useEffect(() => {
    if (recommendation?.message_content) {
      setEditedMessage(recommendation.message_content);
    }
  }, [recommendation]);
  
  // Initialize edited message on component mount
  useEffect(() => {
    if (recommendation?.message_content && !editedMessage) {
      setEditedMessage(recommendation.message_content);
    }
  }, [recommendation?.message_content, editedMessage]);

  const getChannelIcon = (channel) => {
    switch (channel) {
      case 'email':
        return <Mail className="h-5 w-5" />;
      case 'phone':
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
      case 'phone':
        return 'Phone';
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
      case 'phone':
        return 'bg-blue-100 text-blue-800';
      case 'newsletter':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-blue-100 text-blue-800';
    }
  };

  const handleSendMessage = async () => {
    const message = isEditing ? editedMessage : recommendation.message_content;
    
    // All leads use email - check for email address
    if (!lead.lead_data?.email) {
      toast.error('No email address found for this lead');
      return;
    }

    setIsSending(true);
    
    try {
      // All leads send RAG-personalized emails
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

      // Send RAG-personalized email via API
      const result = await leadAPI.sendEmail(
        lead.lead_id,
        lead.lead_data.email,
        leadData,
        emailType  // Always "rag" for RAG personalization
      );

      toast.success(`Email sent successfully!`);
      
      console.log('Message sent:', result);
      
      // Close the panel after successful send
      setTimeout(() => {
        onClose();
      }, 1500);
      
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error(`Failed to send RAG-personalized email. Please check your configuration.`);
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
                  {mode === 'view' ? 'Lead Engagement Details' : 'RAG-Personalized Email Preview'}
                </h2>
                <p className="text-sm text-gray-500">
                  {mode === 'view' 
                    ? 'View detailed engagement metrics and behavioral data' 
                    : `AI-powered personalized email with ${lead.score?.heat_score === 'hot' ? 'urgent' : lead.score?.heat_score === 'warm' ? 'nurturing' : 'educational'} tone based on lead heat score`
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
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
              <Mail className="h-4 w-4" />
              <span className="ml-2">RAG-Personalized Email</span>
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
              <span className="ml-2">{lead.score?.heat_score?.toUpperCase() || 'HOT'} Lead</span>
            </span>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800">
              <span className="ml-2">{lead.score?.heat_score === 'hot' ? 'Urgent' : lead.score?.heat_score === 'warm' ? 'Nurturing' : 'Educational'} Tone</span>
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
                      Generating RAG-personalized email with {lead.score?.heat_score === 'hot' ? 'urgent' : lead.score?.heat_score === 'warm' ? 'nurturing' : 'educational'} tone...
                    </span>
                  </div>
                ) : personalizedEmail ? (
                  <div className="space-y-4">
                    {/* Email Subject */}
                    <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-400">
                      <div className="text-sm font-medium text-blue-900 mb-1">Subject Line:</div>
                      <div className="text-lg font-semibold text-blue-800">
                        {cleanSubjectLine(personalizedEmail.subject, lead.score?.heat_score || 'hot', lead.lead_data)}
                      </div>
                      {!personalizedEmail.subject && (
                        <div className="text-xs text-blue-600 mt-1">
                          (Using fallback subject)
                        </div>
                      )}
                    </div>
                    
                    {/* Generated Email Content */}
                    <div className="bg-gray-50 rounded-lg p-4 border">
                      <div className="text-sm font-medium text-gray-700 mb-2">
                        📧 Generated Email Content:
                      </div>
                      
                      <div className="bg-white rounded-lg border overflow-hidden shadow-lg">
                        <style>
                          {`
                            .email-preview {
                              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                              line-height: 1.7;
                              font-size: 16px;
                              color: #333;
                            }
                            .email-preview strong {
                              font-weight: bold !important;
                              color: #1f2937 !important;
                            }
                            .email-preview br {
                              line-height: 1.5;
                            }
                            .email-preview del {
                              text-decoration: line-through;
                              color: #ef4444 !important;
                              opacity: 0.8;
                            }
                          `}
                        </style>
                        {personalizedEmail.content ? (
                          <div 
                            className="email-preview p-4 bg-white"
                            dangerouslySetInnerHTML={{ __html: personalizedEmail.content }}
                          />
                        ) : personalizedEmail.html_content ? (
                          <div 
                            className="email-preview p-4 bg-white"
                            dangerouslySetInnerHTML={{ __html: personalizedEmail.html_content }}
                          />
                        ) : (
                          <div className="p-4 text-gray-500 text-center">
                            No email content available
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
                      {recommendation.message_content}
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
                      {recommendation.message_content}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

        </div>

        {/* Actions */}
        {mode === 'message' && (
          <div className="p-6 border-t border-gray-200 bg-gray-50">
            
            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={handleSendMessage}
                disabled={isSending}
                className="btn btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Mail className="h-4 w-4" />
                <span>
                  {isSending ? 'Sending...' : 'Send RAG-Personalized Email'}
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