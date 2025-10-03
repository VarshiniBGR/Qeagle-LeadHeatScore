import axios from 'axios';

const API_BASE_URL = '/api/v1';

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 15000,  // Reduced from 30s to 15s
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor
api.interceptors.request.use(
    (config) => {
        // Add request timestamp
        config.metadata = { startTime: new Date() };
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor
api.interceptors.response.use(
    (response) => {
        // Log response time
        const endTime = new Date();
        const duration = endTime - response.config.metadata.startTime;
        console.log(`API call to ${response.config.url} took ${duration}ms`);
        return response;
    },
    (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
    }
);

// Lead scoring API
export const leadAPI = {
    // Score a single lead
    scoreLead: async (leadData) => {
        const response = await api.post('/score', leadData);
        return response.data;
    },

    // Get recommendation for a lead
    getRecommendation: async (leadData) => {
        const response = await api.post('/recommend', leadData);
        return response.data;
    },

    // Upload CSV file (optimized - should be very fast now)
    uploadCSV: async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await api.post('/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            timeout: 10000  // Reduced to 10 seconds - should be much faster now
        });
        return response.data;
    },

    // Batch score leads (increased timeout for processing)
    batchScoreLeads: async (leadsData) => {
        const response = await api.post('/batch-score', leadsData, {
            timeout: 120000  // 2 minutes timeout for batch processing
        });
        return response.data;
    },

    // Get leads (increased timeout for processing)
    getLeads: async (limit = 100, offset = 0) => {
        const response = await api.get('/leads', {
            params: { limit, offset },
            timeout: 30000  // Increased to 30 seconds for leads processing
        });
        return response.data;
    },

    // Get specific lead
    getLead: async (leadId) => {
        const response = await api.get(`/leads/${leadId}`);
        return response.data;
    },

    // Clear all leads
    clearAllLeads: async () => {
        const response = await api.delete('/leads/clear');
        return response.data;
    },

    // Generate recommendation for a specific lead
    generateLeadRecommendation: async (leadId) => {
        const response = await api.post(`/leads/${leadId}/recommendation`);
        return response.data;
    },

    // Get personalized email content without sending
    getPersonalizedEmail: async (leadData) => {
        const response = await api.post('/get-personalized-email', leadData, {
            timeout: 10000  // 10 second timeout for RAG email generation
        });
        return response.data;
    },

    // Send email to lead
    sendEmail: async (leadId, toEmail, leadData, emailType = "template") => {
        const response = await api.post('/send-email', {
            lead_id: leadId,
            to_email: toEmail,
            lead_data: leadData,
            email_type: emailType  // "template" or "rag"
        });
        return response.data;
    },

    // Send Telegram message to phone number
    sendTelegramMessage: async (phoneNumber, message, leadData) => {
        const response = await api.post('/send-telegram-to-phone', {
            phone_number: phoneNumber,
            message: message,
            lead_data: leadData
        });
        return response.data;
    },

    // Send WhatsApp message to phone number
    sendWhatsAppMessage: async (phoneNumber, message) => {
        const response = await api.post('/send-whatsapp-message', {
            phone_number: phoneNumber,
            message: message
        });
        return response.data;
    },
};

// System API
export const systemAPI = {
    // Health check (use proxy for consistency)
    healthCheck: async () => {
        const response = await api.get('/health', {
            timeout: 10000  // Increased to 10 seconds for health check
        });
        return response.data;
    },

    // Get metrics
    getMetrics: async () => {
        const response = await api.get('/metrics');
        return response.data;
    },

    // Train model
    trainModel: async (csvPath = 'data/leads.csv') => {
        const response = await api.post('/train', { csv_path: csvPath });
        return response.data;
    },

    // Ingest knowledge documents
    ingestKnowledge: async () => {
        const response = await api.post('/ingest-knowledge');
        return response.data;
    },

    // Send test email
    sendTestEmail: async (toEmail) => {
        const response = await api.post('/send-test-email', null, {
            params: { to_email: toEmail }
        });
        return response.data;
    },
};

// A/B Testing API
export const abTestingAPI = {
    // Generate A/B test messages
    generateABTestMessage: async (leadData) => {
        const response = await api.post('/ab-test-message', leadData);
        return response.data;
    },

    // Submit A/B test evaluation
    submitABEvaluation: async (evaluationData) => {
        const response = await api.post('/submit-ab-evaluation', evaluationData);
        return response.data;
    },
};
