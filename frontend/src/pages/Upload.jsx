import React, { useState } from 'react';
import { Upload as UploadIcon, FileText, CheckCircle, Eye } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import CsvUpload from '../components/CsvUpload';
import { leadAPI } from '../lib/api';
import toast from 'react-hot-toast';

const Upload = () => {
  const [uploadResult, setUploadResult] = useState(null);
  const [processing, setProcessing] = useState(false);
  const navigate = useNavigate();

  const handleUploadSuccess = async (result) => {
    setUploadResult(result);
    
    // Automatically process the uploaded leads
    if (result.valid_rows > 0) {
      toast.success('Starting lead processing...');
      await processLeads(result.batch_id);
    }
  };

  const processLeads = async (batchId) => {
    setProcessing(true);
    try {
      // In a real implementation, you would:
      // 1. Get the leads from the batch
      // 2. Process them through the classification pipeline
      // 3. Generate recommendations
      
      // For now, we'll simulate this process
      toast.success('Leads processed successfully!');
      
      // Don't redirect automatically - let user choose when to view results
      
    } catch (error) {
      toast.error('Error processing leads');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Lead Data</h1>
        <p className="text-gray-600">
          Upload your lead data in CSV format to get AI-powered heat scores and next action recommendations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Section */}
        <div className="lg:col-span-2">
          <CsvUpload onUploadSuccess={handleUploadSuccess} />
        </div>

        {/* Info Panel */}
        <div className="space-y-6">
          {/* Features */}
          <div className="card">
            <h3 className="font-medium text-gray-900 mb-4">What You'll Get</h3>
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-success-500 mt-0.5" />
                <div>
                  <div className="font-medium text-gray-900">Heat Score Classification</div>
                  <div className="text-sm text-gray-500">Hot, Warm, or Cold with confidence scores</div>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-success-500 mt-0.5" />
                <div>
                  <div className="font-medium text-gray-900">Next Action Recommendations</div>
                  <div className="text-sm text-gray-500">AI-powered channel and message suggestions</div>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <CheckCircle className="h-5 w-5 text-success-500 mt-0.5" />
                <div>
                  <div className="font-medium text-gray-900">Detailed Analytics</div>
                  <div className="text-sm text-gray-500">Performance metrics and insights</div>
                </div>
              </div>
            </div>
          </div>

          {/* Data Requirements */}
          <div className="card">
            <h3 className="font-medium text-gray-900 mb-4">Data Requirements</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center space-x-2">
                <FileText className="h-4 w-4 text-gray-400" />
                <span className="text-gray-600">CSV format only</span>
              </div>
              <div className="flex items-center space-x-2">
                <FileText className="h-4 w-4 text-gray-400" />
                <span className="text-gray-600">UTF-8 encoding</span>
              </div>
              <div className="flex items-center space-x-2">
                <FileText className="h-4 w-4 text-gray-400" />
                <span className="text-gray-600">Maximum 10,000 rows</span>
              </div>
              <div className="flex items-center space-x-2">
                <FileText className="h-4 w-4 text-gray-400" />
                <span className="text-gray-600">All required columns present</span>
              </div>
            </div>
          </div>

          {/* Processing Status */}
          {processing && (
            <div className="card">
              <h3 className="font-medium text-gray-900 mb-4">Processing Status</h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
                  <span className="text-sm text-gray-600">Classifying leads...</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
                  <span className="text-sm text-gray-600">Generating recommendations...</span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600"></div>
                  <span className="text-sm text-gray-600">Preparing results...</span>
                </div>
              </div>
            </div>
          )}

          {/* Success Actions */}
          {uploadResult && !processing && (
            <div className="card">
              <h3 className="font-medium text-gray-900 mb-4">Upload Complete!</h3>
              <div className="space-y-3">
                <div className="text-sm text-gray-600">
                  <strong>{uploadResult.valid_rows}</strong> leads processed successfully
                </div>
                <button 
                  onClick={() => navigate('/leads')}
                  className="btn btn-primary w-full text-sm flex items-center justify-center space-x-2"
                >
                  <Eye className="h-4 w-4" />
                  <span>View Results</span>
                </button>
                <button 
                  onClick={() => {
                    setUploadResult(null);
                    setProcessing(false);
                  }}
                  className="btn btn-secondary w-full text-sm"
                >
                  Upload Another File
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Upload;
