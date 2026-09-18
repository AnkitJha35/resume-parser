import React, { useState, useEffect } from 'react';
import type {
  BenchmarkSummary,
  ConfigResponse,
  DiagnosticResponse,
  FixtureItem,
  HealthResponse,
  ParseResponse,
} from './types';
import {
  getConfig,
  getFixtures,
  getHealth,
  parseDiagnostic,
  parseFixture,
  parseResume,
  runBenchmark,
} from './api';
import { Header } from './components/Header';
import { UploadPanel } from './components/UploadPanel';
import { ResultSummary } from './components/ResultSummary';
import { ResumeViewer } from './components/ResumeViewer';
import { ProvenanceView } from './components/ProvenanceView';
import { StructureDebugView } from './components/StructureDebugView';
import { BenchmarkView } from './components/BenchmarkView';
import {
  FileText,
  ShieldCheck,
  Table,
  BarChart3,
  AlertCircle,
  AlertTriangle,
  X,
} from 'lucide-react';

type MainView = 'resume' | 'provenance' | 'structure' | 'benchmark';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [fixtures, setFixtures] = useState<FixtureItem[]>([]);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [error, setError] = useState<string | null>(null);

  const [parseResponse, setParseResponse] = useState<ParseResponse | null>(null);
  const [diagnosticResponse, setDiagnosticResponse] = useState<DiagnosticResponse | null>(null);
  const [benchmarkSummary, setBenchmarkSummary] = useState<BenchmarkSummary | null>(null);
  const [isBenchmarkLoading, setIsBenchmarkLoading] = useState(false);

  const [activeView, setActiveView] = useState<MainView>('resume');

  // Load health, config, and fixtures on mount
  useEffect(() => {
    getHealth()
      .then((h) => setHealth(h))
      .catch((err) => console.warn('Failed to fetch health:', err));

    getConfig()
      .then((c) => setConfig(c))
      .catch((err) => console.warn('Failed to fetch config:', err));

    getFixtures()
      .then((f) => setFixtures(f))
      .catch((err) => console.warn('Failed to fetch fixtures:', err));
  }, []);

  const handleParse = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setLoadingMessage('Parsing resume with semantic LLM pipeline (1 request)...');
    setError(null);

    try {
      const res = await parseResume(selectedFile);
      setParseResponse(res);
      setActiveView('resume');
    } catch (err: any) {
      setError(err.message || 'Failed to parse resume');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  const handleDiagnostic = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setLoadingMessage('Extracting Document IR layout & table structures...');
    setError(null);

    try {
      const res = await parseDiagnostic(selectedFile);
      setDiagnosticResponse(res);
      setActiveView('structure');
    } catch (err: any) {
      setError(err.message || 'Failed to extract diagnostics');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  const handleSelectFixture = async (fixtureId: string) => {
    setIsLoading(true);
    setLoadingMessage(`Loading and parsing benchmark fixture: ${fixtureId}...`);
    setError(null);

    try {
      const res = await parseFixture(fixtureId);
      setParseResponse(res);
      setActiveView('resume');
    } catch (err: any) {
      setError(err.message || 'Failed to parse fixture');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  };

  const handleRunBenchmark = async (suite: string) => {
    setIsBenchmarkLoading(true);
    setError(null);

    try {
      const summary = await runBenchmark(suite);
      setBenchmarkSummary(summary);
    } catch (err: any) {
      setError(err.message || 'Failed to run benchmark');
    } finally {
      setIsBenchmarkLoading(false);
    }
  };

  const provenanceRecords = parseResponse?.diagnostics?.provenance || [];

  return (
    <div className="app-layout">
      <Header health={health} config={config} />

      <main className="main-content">
        {/* ERROR BANNER */}
        {error && (
          <div className="error-banner">
            <div className="error-content">
              <AlertCircle size={18} className="error-icon" />
              <div className="error-text">
                <strong>Error Occurred:</strong>
                <pre>{error}</pre>
              </div>
            </div>
            <button className="error-close-btn" onClick={() => setError(null)}>
              <X size={16} />
            </button>
          </div>
        )}

        {/* TOP CONTROLS: UPLOAD PANEL */}
        <UploadPanel
          selectedFile={selectedFile}
          onSelectFile={(f) => {
            setSelectedFile(f);
            setError(null);
          }}
          onParse={handleParse}
          onDiagnostic={handleDiagnostic}
          isLoading={isLoading}
          loadingMessage={loadingMessage}
          fixtures={fixtures}
          onSelectFixture={handleSelectFixture}
        />

        {/* VALIDATION WARNING BANNER */}
        {parseResponse && (parseResponse.status === 'validation_failed' || parseResponse.status === 'VALIDATION_FAILED') && (
          <div className="validation-warning-banner">
            <div className="warning-banner-content">
              <AlertTriangle size={20} className="warning-banner-icon" />
              <div className="warning-banner-text">
                <h4>
                  Semantic Validation Reported {parseResponse.violations?.length || 0} Invariant Violation(s)
                </h4>
                <p>
                  Extraction completed and the extracted Resume is available below for inspection.
                </p>
                {parseResponse.violations && parseResponse.violations.length > 0 && (
                  <ul className="violations-inline-list">
                    {parseResponse.violations.map((v, i) => (
                      <li key={i}>{v}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        )}

        {/* SUMMARY BAR */}
        {parseResponse && (
          <ResultSummary
            success={parseResponse.success}
            status={parseResponse.status}
            violationsCount={parseResponse.violations?.length || 0}
            metadata={parseResponse.metadata}
            totalBlocks={parseResponse.diagnostics?.totalBlocks}
            totalTokens={parseResponse.diagnostics?.tokenUsage?.totalTokens}
            archetype={parseResponse.diagnostics?.archetype}
          />
        )}

        {/* MAIN VIEW NAVIGATION */}
        <div className="view-switcher">
          <button
            className={`view-tab ${activeView === 'resume' ? 'active' : ''}`}
            onClick={() => setActiveView('resume')}
          >
            <FileText size={16} />
            <span>Canonical Resume</span>
            {parseResponse && (
              <span
                className={`tab-pill ${
                  parseResponse.status === 'validation_failed'
                    ? 'pill-warning'
                    : 'tab-pill-ready'
                }`}
              >
                {parseResponse.status === 'validation_failed'
                  ? 'Validation Failed'
                  : 'Ready'}
              </span>
            )}
          </button>

          <button
            className={`view-tab ${activeView === 'provenance' ? 'active' : ''}`}
            onClick={() => setActiveView('provenance')}
          >
            <ShieldCheck size={16} />
            <span>Provenance & Evidence</span>
            {provenanceRecords.length > 0 && (
              <span className="tab-pill pill-accent">{provenanceRecords.length}</span>
            )}
          </button>

          <button
            className={`view-tab ${activeView === 'structure' ? 'active' : ''}`}
            onClick={() => setActiveView('structure')}
          >
            <Table size={16} />
            <span>Structure & Tables Debug</span>
            {diagnosticResponse && (
              <span className="tab-pill pill-accent">
                {diagnosticResponse.block_count} blocks
              </span>
            )}
          </button>

          <button
            className={`view-tab ${activeView === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveView('benchmark')}
          >
            <BarChart3 size={16} />
            <span>Benchmark Evaluation</span>
            {benchmarkSummary && (
              <span className="tab-pill pill-accent">
                {benchmarkSummary.hard_correctness_pass_rate_pct.toFixed(0)}%
              </span>
            )}
          </button>
        </div>

        {/* ACTIVE VIEW DISPLAY */}
        <div className="view-container">
          {activeView === 'resume' && (
            <>
              {parseResponse && parseResponse.resume ? (
                <ResumeViewer resume={parseResponse.resume} />
              ) : (
                <div className="empty-view-state card">
                  <FileText size={40} className="text-muted" />
                  <h3>No Parsed Resume Loaded</h3>
                  <p>
                    Select or upload a PDF above and click <strong>Parse Resume</strong>, or select a
                    fixture from the dropdown.
                  </p>
                </div>
              )}
            </>
          )}

          {activeView === 'provenance' && (
            <>
              {provenanceRecords.length > 0 ? (
                <ProvenanceView
                  records={provenanceRecords}
                  archetype={parseResponse?.diagnostics?.archetype}
                  totalBlocks={parseResponse?.diagnostics?.totalBlocks}
                />
              ) : (
                <div className="empty-view-state card">
                  <ShieldCheck size={40} className="text-muted" />
                  <h3>No Provenance Records Available</h3>
                  <p>
                    Run <strong>Parse Resume</strong> on a document to inspect field-level source
                    block provenance.
                  </p>
                </div>
              )}
            </>
          )}

          {activeView === 'structure' && (
            <>
              {diagnosticResponse ? (
                <StructureDebugView diagnostic={diagnosticResponse} />
              ) : (
                <div className="empty-view-state card">
                  <Table size={40} className="text-muted" />
                  <h3>No Structural Diagnostic Data</h3>
                  <p>
                    Click <strong>Diagnostic Parse (Structural Only)</strong> on an uploaded PDF to
                    inspect the raw layout blocks, visual reading orders, and table bindings.
                  </p>
                </div>
              )}
            </>
          )}

          {activeView === 'benchmark' && (
            <BenchmarkView
              summary={benchmarkSummary}
              onRunBenchmark={handleRunBenchmark}
              isLoading={isBenchmarkLoading}
            />
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
