import React, { useState } from 'react';
import type { GenericDocumentJson, GenericDocumentSectionJson, JsonFidelityReport } from '../types';
import { Copy, Check, Download, FileJson, ChevronDown, ChevronRight, Layers, ShieldCheck, AlertTriangle } from 'lucide-react';
import './GenericJsonViewer.css';

interface GenericJsonViewerProps {
  documentJson: GenericDocumentJson | Record<string, any>;
  documentProvenance?: GenericDocumentJson | Record<string, any> | null;
  jsonFidelity?: JsonFidelityReport | null;
}

export const GenericJsonViewer: React.FC<GenericJsonViewerProps> = ({
  documentJson,
  documentProvenance,
  jsonFidelity,
}) => {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<'structured' | 'data_raw' | 'provenance'>('structured');
  const [collapsedSections, setCollapsedSections] = useState<Record<number, boolean>>({});
  const [showIssues, setShowIssues] = useState(false);

  // Clean data JSON stringified for copying, downloading, and raw view
  const formattedDataJson = JSON.stringify(documentJson, null, 2);
  const formattedProvenanceJson = documentProvenance
    ? JSON.stringify(documentProvenance, null, 2)
    : '';

  const doc = (documentJson as GenericDocumentJson)?.document || documentJson;
  const sections: GenericDocumentSectionJson[] = doc?.sections || [];
  const pageCount: number = doc?.pages || 1;

  const totalBlocks = sections.reduce((acc, s) => acc + (s.blocks?.length || 0), 0);
  const totalTables = sections.reduce(
    (acc, s) => acc + (s.blocks?.filter((b) => b.type === 'table').length || 0),
    0
  );

  const handleCopy = () => {
    // Copy strictly Clean Data JSON
    navigator.clipboard.writeText(formattedDataJson).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleDownload = () => {
    // Download strictly Clean Data JSON
    const blob = new Blob([formattedDataJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'document_clean_data.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleSection = (idx: number) => {
    setCollapsedSections((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  const expandAll = () => setCollapsedSections({});
  const collapseAll = () => {
    const all: Record<number, boolean> = {};
    sections.forEach((_, idx) => {
      all[idx] = true;
    });
    setCollapsedSections(all);
  };

  return (
    <div className="gjv-root">
      {/* TOOLBAR */}
      <div className="gjv-toolbar">
        <div className="gjv-toolbar-left">
          <div className="gjv-title-badge">
            <span className="gjv-icon">
              <FileJson size={18} />
            </span>
            <span className="gjv-title">Clean Document JSON</span>
          </div>
          <span className="gjv-pill gjv-pill-info">{pageCount} {pageCount === 1 ? 'Page' : 'Pages'}</span>
          <span className="gjv-pill gjv-pill-accent">{sections.length} Sections</span>
          <span className="gjv-pill gjv-pill-neutral">{totalBlocks} Blocks</span>
          {totalTables > 0 && (
            <span className="gjv-pill gjv-pill-table">{totalTables} Tables</span>
          )}
        </div>

        <div className="gjv-toolbar-right">
          <div className="gjv-mode-toggle">
            <button
              className={`gjv-btn-toggle ${viewMode === 'structured' ? 'active' : ''}`}
              onClick={() => setViewMode('structured')}
              title="Structured section cards view"
            >
              <Layers size={14} />
              <span>Structured View</span>
            </button>
            <button
              className={`gjv-btn-toggle ${viewMode === 'data_raw' ? 'active' : ''}`}
              onClick={() => setViewMode('data_raw')}
              title="Clean user-facing Data JSON (without parser internals)"
            >
              <FileJson size={14} />
              <span>Data JSON</span>
            </button>
            {documentProvenance && (
              <button
                className={`gjv-btn-toggle ${viewMode === 'provenance' ? 'active' : ''}`}
                onClick={() => setViewMode('provenance')}
                title="Internal JSON with block IDs, provenance, and geometry"
              >
                <ShieldCheck size={14} />
                <span>Debug / Provenance</span>
              </button>
            )}
          </div>

          <button className="gjv-btn gjv-btn-secondary" onClick={handleDownload} title="Download Clean Data JSON file">
            <Download size={14} />
            <span>Download Data JSON</span>
          </button>

          <button
            className={`gjv-btn ${copied ? 'gjv-btn-success' : 'gjv-btn-primary'}`}
            onClick={handleCopy}
            title="Copy Clean Data JSON to clipboard"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            <span>{copied ? 'Copied!' : 'Copy Data JSON'}</span>
          </button>
        </div>
      </div>

      {/* FIDELITY DIAGNOSTICS CARD */}
      {jsonFidelity && (
        <div className={`gjv-fidelity-card ${jsonFidelity.valid ? 'fidelity-pass' : 'fidelity-fail'}`}>
          <div className="gjv-fidelity-header">
            <div className="gjv-fidelity-title-group">
              {jsonFidelity.valid ? (
                <ShieldCheck size={18} className="gjv-fidelity-status-icon text-success" />
              ) : (
                <AlertTriangle size={18} className="gjv-fidelity-status-icon text-danger" />
              )}
              <span className="gjv-fidelity-title">JSON Fidelity</span>
              <span className={`gjv-fidelity-badge ${jsonFidelity.valid ? 'badge-pass' : 'badge-fail'}`}>
                {jsonFidelity.valid ? '✓ PASS' : '⚠ FAIL'}
              </span>
            </div>
            {!jsonFidelity.valid && jsonFidelity.issues && jsonFidelity.issues.length > 0 && (
              <button
                className="gjv-btn-issues-toggle"
                onClick={() => setShowIssues(!showIssues)}
              >
                {showIssues ? 'Hide Issues' : `Show Issues (${jsonFidelity.issues.length})`}
              </button>
            )}
          </div>

          <div className="gjv-fidelity-grid">
            <div className="gjv-fidelity-stat">
              <span className="stat-label">Source blocks</span>
              <span className="stat-value">{jsonFidelity.source_blocks}</span>
            </div>
            <div className="gjv-fidelity-stat">
              <span className="stat-label">JSON ownership</span>
              <span className="stat-value">{jsonFidelity.json_owned_blocks}</span>
            </div>
            <div className={`gjv-fidelity-stat ${jsonFidelity.missing_blocks > 0 ? 'stat-error' : ''}`}>
              <span className="stat-label">Missing</span>
              <span className="stat-value">{jsonFidelity.missing_blocks}</span>
            </div>
            <div className={`gjv-fidelity-stat ${jsonFidelity.duplicate_blocks > 0 ? 'stat-error' : ''}`}>
              <span className="stat-label">Duplicates</span>
              <span className="stat-value">{jsonFidelity.duplicate_blocks}</span>
            </div>
            <div className={`gjv-fidelity-stat ${jsonFidelity.unsupported_blocks > 0 ? 'stat-error' : ''}`}>
              <span className="stat-label">Unsupported</span>
              <span className="stat-value">{jsonFidelity.unsupported_blocks}</span>
            </div>
            <div className={`gjv-fidelity-stat ${jsonFidelity.provenance_errors > 0 ? 'stat-error' : ''}`}>
              <span className="stat-label">Provenance errors</span>
              <span className="stat-value">{jsonFidelity.provenance_errors}</span>
            </div>
          </div>

          {showIssues && jsonFidelity.issues && jsonFidelity.issues.length > 0 && (
            <div className="gjv-fidelity-issues-list">
              <div className="gjv-fidelity-issues-header">Detected Fidelity Issues:</div>
              <ul className="gjv-fidelity-issues-ul">
                {jsonFidelity.issues.map((issue, idx) => (
                  <li key={idx} className="gjv-fidelity-issue-item">
                    <code>{issue}</code>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* CONTENT AREA */}
      <div className="gjv-content">
        {viewMode === 'structured' ? (
          <div className="gjv-structured-container">
            <div className="gjv-section-controls">
              <span className="gjv-controls-label">Interactive Section Explorer</span>
              <div className="gjv-controls-actions">
                <button className="gjv-link-btn" onClick={expandAll}>
                  Expand All
                </button>
                <span className="gjv-divider">•</span>
                <button className="gjv-link-btn" onClick={collapseAll}>
                  Collapse All
                </button>
              </div>
            </div>

            <div className="gjv-sections-list">
              {sections.map((section, sIdx) => {
                const isCollapsed = collapsedSections[sIdx] || false;
                const blockCount = section.blocks?.length || 0;
                const tableCount = section.blocks?.filter((b) => b.type === 'table').length || 0;

                return (
                  <div key={sIdx} className="gjv-section-card">
                    <div
                      className="gjv-section-header"
                      onClick={() => toggleSection(sIdx)}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="gjv-section-header-left">
                        <span className="gjv-chevron">
                          {isCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                        </span>
                        <span className="gjv-section-index">{sIdx + 1}.</span>
                        <span className="gjv-section-heading">
                          {section.heading ? section.heading : <em className="text-muted">Unheaded Content</em>}
                        </span>
                        <span className="gjv-badge-level">L{section.level || 1}</span>
                      </div>
                      <div className="gjv-section-header-right">
                        <span className="gjv-badge-count">{blockCount} blocks</span>
                        {tableCount > 0 && (
                          <span className="gjv-badge-table">{tableCount} tbl</span>
                        )}
                      </div>
                    </div>

                    {!isCollapsed && (
                      <div className="gjv-section-body">
                        {/* Section JSON snippet */}
                        <pre className="gjv-section-json">
                          <code>{JSON.stringify(section, null, 2)}</code>
                        </pre>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : viewMode === 'data_raw' ? (
          <div className="gjv-raw-container">
            <div className="gjv-raw-header-banner">
              <span className="gjv-raw-badge gjv-raw-badge-clean">Clean Extracted Data (User-Facing)</span>
              <span className="text-muted">Zero parser metadata, zero block IDs, zero bounding boxes</span>
            </div>
            <pre className="gjv-raw-pre">
              <code>{formattedDataJson}</code>
            </pre>
          </div>
        ) : (
          <div className="gjv-raw-container">
            <div className="gjv-raw-header-banner">
              <span className="gjv-raw-badge gjv-raw-badge-provenance">Parser Provenance & Debug Structure</span>
              <span className="text-muted">Contains source block IDs, geometry, page numbers, and item details</span>
            </div>
            <pre className="gjv-raw-pre">
              <code>{formattedProvenanceJson}</code>
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
