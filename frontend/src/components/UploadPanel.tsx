import React, { useRef, useState } from 'react';
import type { FixtureItem } from '../types';
import { UploadCloud, File, Play, Wrench, RefreshCw, X } from 'lucide-react';

interface UploadPanelProps {
  selectedFile: File | null;
  onSelectFile: (file: File | null) => void;
  onParse: () => void;
  onDiagnostic: () => void;
  isLoading: boolean;
  loadingMessage: string;
  fixtures: FixtureItem[];
  onSelectFixture: (fixtureId: string) => void;
}

export const UploadPanel: React.FC<UploadPanelProps> = ({
  selectedFile,
  onSelectFile,
  onParse,
  onDiagnostic,
  isLoading,
  loadingMessage,
  fixtures,
  onSelectFixture,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.pdf')) {
        onSelectFile(file);
      } else {
        alert('Please select a valid PDF file.');
      }
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onSelectFile(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="upload-panel card">
      <div className="upload-header">
        <h3>Resume Input</h3>
        <div className="fixture-quick-select">
          <label htmlFor="fixture-select">Or load benchmark fixture:</label>
          <select
            id="fixture-select"
            className="select-input"
            defaultValue=""
            disabled={isLoading}
            onChange={(e) => {
              if (e.target.value) {
                onSelectFixture(e.target.value);
                e.target.value = '';
              }
            }}
          >
            <option value="" disabled>
              Select standard benchmark fixture...
            </option>
            <optgroup label="Regression 12 Suite">
              {fixtures
                .filter((f) => f.suite === 'regression_12')
                .map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.filename} ({f.candidate_name || f.archetype})
                  </option>
                ))}
            </optgroup>
            {fixtures.some((f) => f.suite === 'generalization') && (
              <optgroup label="Generalization Suite">
                {fixtures
                  .filter((f) => f.suite === 'generalization')
                  .map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.filename} ({f.target_domain || f.archetype})
                    </option>
                  ))}
              </optgroup>
            )}
          </select>
        </div>
      </div>

      <div
        className={`drop-zone ${isDragOver ? 'drag-over' : ''} ${
          selectedFile ? 'has-file' : ''
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          accept="application/pdf,.pdf"
          style={{ display: 'none' }}
          onChange={handleFileInput}
          disabled={isLoading}
        />

        {selectedFile ? (
          <div className="file-info-container">
            <File size={36} className="file-icon" />
            <div className="file-details">
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">{formatFileSize(selectedFile.size)}</span>
            </div>
            <button
              type="button"
              className="btn-icon-clear"
              disabled={isLoading}
              onClick={(e) => {
                e.stopPropagation();
                onSelectFile(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              title="Remove file"
            >
              <X size={18} />
            </button>
          </div>
        ) : (
          <div className="drop-prompt">
            <UploadCloud size={40} className="upload-icon" />
            <p className="primary-prompt">
              Drag and drop resume PDF here, or <span>browse file</span>
            </p>
            <p className="sub-prompt">PDF files only (maximum 10 MB)</p>
          </div>
        )}
      </div>

      {isLoading && (
        <div className="loading-banner">
          <RefreshCw size={16} className="spin-icon" />
          <span>{loadingMessage || 'Processing pipeline...'}</span>
        </div>
      )}

      <div className="panel-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!selectedFile || isLoading}
          onClick={onParse}
        >
          <Play size={16} />
          <span>Parse Resume (Semantic LLM)</span>
        </button>

        <button
          type="button"
          className="btn btn-secondary"
          disabled={!selectedFile || isLoading}
          onClick={onDiagnostic}
        >
          <Wrench size={16} />
          <span>Diagnostic Parse (Structural Only)</span>
        </button>

        {selectedFile && (
          <button
            type="button"
            className="btn btn-ghost"
            disabled={isLoading}
            onClick={() => {
              onSelectFile(null);
              if (fileInputRef.current) fileInputRef.current.value = '';
            }}
          >
            Clear
          </button>
        )}
      </div>
    </div>
  );
};
