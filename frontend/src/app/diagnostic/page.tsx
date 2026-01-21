'use client';

import React, { useState, useCallback } from 'react';
import { Navigation } from '@/components/Navigation';
import { Card, CardContent, CardHeader } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Alert } from '@/components/ui/Alert';
import { useDiagnostic } from '@/hooks/useApi';
import type { DiagnosticResponse, DiagnosticFinding } from '@/types/api';

type ImageType = 'cxr' | 'ct' | 'mri' | 'pathology';

const imageTypeLabels: Record<ImageType, string> = {
  cxr: 'Chest X-Ray',
  ct: 'CT Scan',
  mri: 'MRI',
  pathology: 'Pathology Slide',
};

const severityVariants: Record<string, 'emergency' | 'urgent' | 'info' | 'default' | 'success'> = {
  critical: 'emergency',
  high: 'urgent',
  medium: 'info',
  low: 'success',
};

function FindingCard({ finding, index }: { finding: DiagnosticFinding; index: number }) {
  const confidencePercent = Math.round(finding.confidence * 100);

  return (
    <Card className="mb-3">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-500">Finding #{index + 1}</span>
            {finding.severity && (
              <Badge variant={severityVariants[finding.severity] || 'default'}>
                {finding.severity}
              </Badge>
            )}
          </div>
          <span className={`text-sm font-medium ${confidencePercent < 70 ? 'text-emergency-600' :
              confidencePercent < 85 ? 'text-urgent-600' : 'text-success-600'
            }`}>
            {confidencePercent}% confidence
          </span>
        </div>
        <p className="text-gray-900 font-medium">{finding.finding}</p>
        {finding.location && (
          <p className="text-sm text-gray-500 mt-1">Location: {finding.location}</p>
        )}
      </CardContent>
    </Card>
  );
}

export default function DiagnosticPage() {
  const { analyzeImage, loading, error } = useDiagnostic();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageType, setImageType] = useState<ImageType>('cxr');
  const [clinicalContext, setClinicalContext] = useState('');
  const [result, setResult] = useState<DiagnosticResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    // Convert file to base64
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = (reader.result as string).split(',')[1];

      const response = await analyzeImage({
        image_base64: base64,
        image_type: imageType,
        clinical_context: clinicalContext || undefined,
      });

      if (response) {
        setResult(response);
      }
    };
    reader.readAsDataURL(selectedFile);
  }, [selectedFile, imageType, clinicalContext, analyzeImage]);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setClinicalContext('');
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Medical Image Analysis</h1>
          <p className="text-gray-600 mt-1">
            Upload medical images for AI-powered diagnostic analysis using CXR Foundation and Path Foundation models.
          </p>
        </div>

        {/* Disclaimer */}
        <Alert variant="warning" className="mb-6">
          <strong>Important:</strong> This tool is for demonstration purposes only.
          All AI-generated findings must be reviewed by qualified healthcare professionals
          before any clinical decisions are made.
        </Alert>

        {error && (
          <Alert variant="error" className="mb-6">
            {error}
          </Alert>
        )}

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Panel */}
          <Card>
            <CardHeader>
              <h2 className="text-lg font-semibold text-gray-900">Upload Image</h2>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Image Type Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Image Type
                </label>
                <div className="flex flex-wrap gap-2">
                  {(Object.keys(imageTypeLabels) as ImageType[]).map((type) => (
                    <Button
                      key={type}
                      size="sm"
                      variant={imageType === type ? 'default' : 'outline'}
                      onClick={() => setImageType(type)}
                    >
                      {imageTypeLabels[type]}
                    </Button>
                  ))}
                </div>
              </div>

              {/* File Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Image
                </label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors">
                  {previewUrl ? (
                    <div className="relative">
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="max-h-64 mx-auto rounded-lg"
                      />
                      <button
                        onClick={handleClear}
                        className="absolute top-2 right-2 bg-white rounded-full p-1 shadow-md hover:bg-gray-100"
                      >
                        <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ) : (
                    <div>
                      <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      <p className="mt-2 text-sm text-gray-600">
                        Click to upload or drag and drop
                      </p>
                      <p className="text-xs text-gray-500">
                        DICOM, PNG, JPG up to 10MB
                      </p>
                    </div>
                  )}
                  <input
                    type="file"
                    accept="image/*,.dcm"
                    onChange={handleFileSelect}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                </div>
              </div>

              {/* Clinical Context */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Clinical Context (Optional)
                </label>
                <textarea
                  value={clinicalContext}
                  onChange={(e) => setClinicalContext(e.target.value)}
                  placeholder="e.g., 65yo male, presenting with shortness of breath and cough..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>

              {/* Analyze Button */}
              <Button
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                loading={loading}
                className="w-full"
              >
                {loading ? 'Analyzing...' : 'Analyze Image'}
              </Button>
            </CardContent>
          </Card>

          {/* Results Panel */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">Analysis Results</h2>
                {result && (
                  <Badge variant={result.requires_review ? 'urgent' : 'success'}>
                    {result.requires_review ? 'Requires Review' : 'Complete'}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {!result ? (
                <div className="text-center py-12 text-gray-500">
                  <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  <p>Upload an image and click Analyze to see results</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Overall Confidence */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-700">Overall Confidence</span>
                      <span className="text-lg font-bold">{Math.round(result.confidence * 100)}%</span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${result.confidence < 0.7 ? 'bg-emergency-500' :
                            result.confidence < 0.85 ? 'bg-urgent-500' : 'bg-success-500'
                          }`}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Findings */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">
                      Findings ({result.findings.length})
                    </h3>
                    {result.findings.length === 0 ? (
                      <p className="text-gray-500 text-sm">No significant findings detected.</p>
                    ) : (
                      result.findings.map((finding, index) => (
                        <FindingCard key={index} finding={finding} index={index} />
                      ))
                    )}
                  </div>

                  {/* AI Report */}
                  {result.report && (
                    <div>
                      <h3 className="text-sm font-medium text-gray-700 mb-2">AI Report</h3>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-900 whitespace-pre-wrap">{result.report}</p>
                      </div>
                    </div>
                  )}

                  {/* Processing Time */}
                  <p className="text-xs text-gray-500 text-right">
                    Processed in {result.processing_time_ms}ms
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
