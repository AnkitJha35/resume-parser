import React from 'react';
import type { ParseMetadata } from '../types';
import { CheckCircle2, AlertTriangle, XCircle, Clock, FileText, Cpu, Hash, Database } from 'lucide-react';

interface ResultSummaryProps {
  success: boolean;
  status?: string;
  violationsCount?: number;
  metadata: ParseMetadata | null;
  totalBlocks?: number;
  totalTokens?: number | null;
  archetype?: string;
}

export const ResultSummary: React.FC<ResultSummaryProps> = ({
  success,
  status = 'success',
  violationsCount = 0,
  metadata,
  totalBlocks,
  totalTokens,
  archetype,
}) => {
  if (!metadata) return null;

  const statusNorm = (status || '').toUpperCase();
  const isValidationFailed = statusNorm === 'VALIDATION_FAILED';
  const isPartial = statusNorm === 'PARTIAL';
  const isExtractionFailed = statusNorm === 'EXTRACTION_FAILED';

  return (
    <div className="summary-bar">
      <div className="summary-card">
        <span className="summary-label">Status</span>
        {isValidationFailed ? (
          <div className="summary-val warning-val" title={`${violationsCount} semantic validation violation(s)`}>
            <AlertTriangle size={16} />
            <span>Validation Failed ({violationsCount})</span>
          </div>
        ) : isPartial ? (
          <div className="summary-val warning-val" title="Resume parsed but some major sections are missing">
            <AlertTriangle size={16} />
            <span>Partially Parsed</span>
          </div>
        ) : success ? (
          <div className="summary-val success-val">
            <CheckCircle2 size={16} />
            <span>Parsed Successfully</span>
          </div>
        ) : isExtractionFailed ? (
          <div className="summary-val error-val">
            <XCircle size={16} />
            <span>Extraction Failed</span>
          </div>
        ) : (
          <div className="summary-val error-val">
            <XCircle size={16} />
            <span>Failed</span>
          </div>
        )}
      </div>

      <div className="summary-card">
        <span className="summary-label">Processing Latency</span>
        <div className="summary-val">
          <Clock size={16} />
          <span>{metadata.latencyMs.toFixed(0)} ms</span>
        </div>
      </div>

      <div className="summary-card">
        <span className="summary-label">Pages</span>
        <div className="summary-val">
          <FileText size={16} />
          <span>{metadata.pageCount} page{metadata.pageCount > 1 ? 's' : ''}</span>
        </div>
      </div>

      <div className="summary-card">
        <span className="summary-label">Provider & Model</span>
        <div className="summary-val">
          <Cpu size={16} />
          <span title={`${metadata.provider} / ${metadata.model}`}>
            {metadata.provider} : {metadata.model.split('/').pop()}
          </span>
        </div>
      </div>

      <div className="summary-card">
        <span className="summary-label">LLM Requests</span>
        <div className="summary-val">
          <Hash size={16} />
          <span>{metadata.requestCount} request</span>
        </div>
      </div>

      {totalBlocks !== undefined && (
        <div className="summary-card">
          <span className="summary-label">Source Blocks</span>
          <div className="summary-val">
            <Database size={16} />
            <span>{totalBlocks} blocks</span>
          </div>
        </div>
      )}

      {archetype && (
        <div className="summary-card">
          <span className="summary-label">Archetype</span>
          <div className="summary-val highlight-val">
            <span>{archetype}</span>
          </div>
        </div>
      )}

      {totalTokens !== undefined && totalTokens !== null && (
        <div className="summary-card">
          <span className="summary-label">Total Tokens</span>
          <div className="summary-val">
            <span>{totalTokens.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
};
