import React, { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, X } from 'lucide-react';
import { leadAPI } from '../lib/api';
import toast from 'react-hot-toast';

const CsvUpload = ({ onUploadSuccess }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    if (!file.name.endsWith('.csv')) {
      toast.error('Please upload a CSV file');
      return;
    }

    setUploading(true);
    setUploadResult(null);

    try {
      const result = await leadAPI.uploadCSV(file);
      setUploadResult(result);
      toast.success(`Successfully uploaded ${result.valid_rows} leads`);
      
      if (onUploadSuccess) {
        onUploadSuccess(result);
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const clearResult = () => {
    setUploadResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-primary-400 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="space-y-4">
          <div className="mx-auto w-12 h-12 text-gray-400">
            <Upload className="w-full h-full" />
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-gray-900">
              Upload Lead Data
            </h3>
            <p className="text-gray-500">
              Drag and drop your CSV file here, or click to browse
            </p>
          </div>
          
          <div className="text-sm text-gray-400">
            <p>Supported format: CSV</p>
            <p>Required columns: name, email, source, recency_days, region, role, campaign, page_views, last_touch, prior_course_interest</p>
            <p>Optional columns: phone, search_keywords, time_spent, course_actions</p>
          </div>
        </div>
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="card">
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              <CheckCircle className="h-6 w-6 text-success-500 mt-1" />
              <div>
                <h3 className="font-medium text-gray-900">Upload Successful</h3>
                <div className="mt-2 space-y-1 text-sm text-gray-600">
                  <p><strong>File:</strong> {uploadResult.filename}</p>
                  <p><strong>Total Rows:</strong> {uploadResult.total_rows}</p>
                  <p><strong>Valid Rows:</strong> {uploadResult.valid_rows}</p>
                  <p><strong>Invalid Rows:</strong> {uploadResult.invalid_rows}</p>
                  <p><strong>Batch ID:</strong> {uploadResult.batch_id}</p>
                </div>
              </div>
            </div>
            
            <button
              onClick={clearResult}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}

      {/* Loading State */}
      {uploading && (
        <div className="card">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
            <span className="text-gray-600">Uploading and validating file...</span>
          </div>
        </div>
      )}

      {/* Sample Data Format */}
      <div className="card">
        <h3 className="font-medium text-gray-900 mb-4">Sample Data Format</h3>
        <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
          <pre className="text-sm text-gray-700">
{`name,email,phone,source,recency_days,region,role,campaign,page_views,last_touch,prior_course_interest,search_keywords,time_spent,course_actions
John Doe,john@example.com,+1-555-0123,website,5,US,Manager,summer_sale,15,email,high,"customer service, sales",300,download_brochure
Jane Smith,jane@company.com,+1-555-0124,linkedin,12,EU,Engineer,tech_conference,8,social,medium,"project management, agile",250,view_course
Bob Johnson,bob@startup.io,+1-555-0125,referral,2,APAC,Director,partner_program,25,phone,high,"leadership, strategy",450,book_demo`}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default CsvUpload;
