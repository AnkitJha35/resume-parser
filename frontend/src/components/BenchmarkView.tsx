import React, { useState } from 'react';
import type { BenchmarkSummary, SemanticParseResult } from '../types';
import {
  Play,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  Zap,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  ShieldAlert,
} from 'lucide-react';

interface BenchmarkViewProps {
  summary: BenchmarkSummary | null;
  onRunBenchmark: (suite: string) => void;
  isLoading: boolean;
}

export const BenchmarkView: React.FC<BenchmarkViewProps> = ({
  summary,
  onRunBenchmark,
  isLoading,
}) => {
  const [suite, setSuite] = useState<string>('regression_12');
  const [expandedFixture, setExpandedFixture] = useState<string | null>(null);

  const toggleFixture = (filename: string) => {
    setExpandedFixture(expandedFixture === filename ? null : filename);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'PASS':
        return (
          <span className="badge badge-pass">
            <CheckCircle2 size={13} /> PASS
          </span>
        );
      case 'PARTIAL':
        return (
          <span className="badge badge-partial">
            <AlertTriangle size={13} /> PARTIAL
          </span>
        );
      case 'VALIDATION_FAILED':
        return (
          <span className="badge badge-fail">
            <XCircle size={13} /> VALIDATION_FAILED
          </span>
        );
      default:
        return (
          <span className="badge badge-err">
            <XCircle size={13} /> {status}
          </span>
        );
    }
  };

  return (
    <div className="benchmark-viewer card">
      <div className="benchmark-header">
        <div>
          <h3>Benchmark Evaluation Runner</h3>
          <p className="benchmark-subtitle">
            Execute full evaluation runs across registered regression or generalization suites and inspect live accuracy KPIs.
          </p>
        </div>

        <div className="benchmark-controls">
          <select
            className="select-input"
            value={suite}
            onChange={(e) => setSuite(e.target.value)}
            disabled={isLoading}
          >
            <option value="regression_12">Suite: regression_12 (12 documents)</option>
            <option value="generalization">Suite: generalization</option>
          </select>

          <button
            className="btn btn-primary"
            onClick={() => onRunBenchmark(suite)}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <RefreshCw size={15} className="spin-icon" />
                <span>Running Suite...</span>
              </>
            ) : (
              <>
                <Play size={15} />
                <span>Run Benchmark</span>
              </>
            )}
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="loading-banner">
          <RefreshCw size={18} className="spin-icon" />
          <div>
            <strong>Evaluating benchmark suite: {suite}</strong>
            <p>Running parser through semantic pipeline and computing evaluation metrics...</p>
          </div>
        </div>
      )}

      {summary && (
        <>
          {/* KPI GRID */}
          <div className="kpi-grid">
            <div className="kpi-card">
              <span className="kpi-label">Hard Correctness</span>
              <span className="kpi-value text-accent">
                {summary.hard_correctness_pass_rate_pct.toFixed(1)}%
              </span>
              <span className="kpi-sub">
                {summary.results.filter((r) => r.hard_correctness_passed).length} / {summary.total_cases} passed
              </span>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">Entity Completeness</span>
              <span className="kpi-value">
                {summary.entity_completeness_rate_pct.toFixed(1)}%
              </span>
              <span className="kpi-sub">Average across entities</span>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">Field Completeness</span>
              <span className="kpi-value">
                {summary.field_completeness_rate_pct.toFixed(1)}%
              </span>
              <span className="kpi-sub">Required core fields</span>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">Skills Recall</span>
              <span className="kpi-value">
                {summary.avg_skills_recall_pct !== null && summary.avg_skills_recall_pct !== undefined
                  ? `${summary.avg_skills_recall_pct.toFixed(1)}%`
                  : '—'}
              </span>
              <span className="kpi-sub">Grounded skill capture</span>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">Requests / Resume</span>
              <span className="kpi-value">
                {summary.avg_requests_per_resume.toFixed(2)}
              </span>
              <span className="kpi-sub">Strict 1-request constraint</span>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">Validation Failures</span>
              <span className={`kpi-value ${summary.validation_failures > 0 ? 'text-fail' : 'text-pass'}`}>
                {summary.validation_failures}
              </span>
              <span className="kpi-sub">Invariant violations</span>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">Total Latency</span>
              <span className="kpi-value">
                {summary.total_elapsed_seconds.toFixed(1)}s
              </span>
              <span className="kpi-sub">
                avg {(summary.total_elapsed_seconds / Math.max(1, summary.total_cases)).toFixed(2)}s / doc
              </span>
            </div>

            {summary.total_tokens && (
              <div className="kpi-card">
                <span className="kpi-label">Total Tokens</span>
                <span className="kpi-value">
                  {summary.total_tokens.toLocaleString()}
                </span>
                <span className="kpi-sub">Prompt + output tokens</span>
              </div>
            )}
          </div>

          {/* PER-FIXTURE RESULTS TABLE */}
          <div className="table-responsive">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th style={{ width: '4%' }}></th>
                  <th style={{ width: '28%' }}>Fixture PDF</th>
                  <th style={{ width: '16%' }}>Archetype</th>
                  <th style={{ width: '14%' }}>Status</th>
                  <th style={{ width: '12%' }}>Hard Correct</th>
                  <th style={{ width: '12%' }}>Violations</th>
                  <th style={{ width: '14%' }}>Latency</th>
                </tr>
              </thead>
              <tbody>
                {summary.results.map((res: SemanticParseResult) => {
                  const isExpanded = expandedFixture === res.filename;
                  const violationCount =
                    (res.validation_violations?.length || 0) +
                    (res.hard_correctness_violations?.length || 0);

                  return (
                    <React.Fragment key={res.filename}>
                      <tr
                        className={`clickable-row ${isExpanded ? 'row-expanded' : ''}`}
                        onClick={() => toggleFixture(res.filename)}
                      >
                        <td>
                          {isExpanded ? (
                            <ChevronDown size={15} />
                          ) : (
                            <ChevronRight size={15} />
                          )}
                        </td>
                        <td>
                          <div className="fixture-title-cell">
                            <strong>{res.filename}</strong>
                            <span className="fixture-domain">{res.target_domain}</span>
                          </div>
                        </td>
                        <td>
                          <span className="badge badge-meta">{res.archetype}</span>
                        </td>
                        <td>{getStatusBadge(res.status)}</td>
                        <td>
                          {res.hard_correctness_passed ? (
                            <span className="badge badge-pass">PASS</span>
                          ) : (
                            <span className="badge badge-fail">FAIL</span>
                          )}
                        </td>
                        <td>
                          {violationCount > 0 ? (
                            <span className="badge badge-violation-count">
                              <ShieldAlert size={12} /> {violationCount}
                            </span>
                          ) : (
                            <span className="badge badge-clean">Clean</span>
                          )}
                        </td>
                        <td>
                          <span className="latency-text">
                            <Clock size={12} /> {res.elapsed_seconds.toFixed(2)}s
                          </span>
                        </td>
                      </tr>

                      {isExpanded && (
                        <tr className="detail-row">
                          <td colSpan={7}>
                            <div className="fixture-details-box">
                              {res.error_message && (
                                <div className="error-alert">
                                  <strong>Error:</strong> {res.error_message}
                                </div>
                              )}

                              {res.validation_violations && res.validation_violations.length > 0 && (
                                <div className="violations-group">
                                  <h4>Validation Violations ({res.validation_violations.length})</h4>
                                  <ul>
                                    {res.validation_violations.map((v, i) => (
                                      <li key={i} className="violation-item">
                                        {v}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}

                              {res.hard_correctness_violations &&
                                res.hard_correctness_violations.length > 0 && (
                                  <div className="violations-group">
                                    <h4>Hard Correctness Discrepancies</h4>
                                    <ul>
                                      {res.hard_correctness_violations.map((v, i) => (
                                        <li key={i} className="violation-item item-hard">
                                          {v}
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                )}

                              {res.notes && (
                                <div className="notes-box">
                                  <strong>Notes:</strong> {res.notes}
                                </div>
                              )}

                              <div className="extracted-quick-summary">
                                <div>
                                  <strong>Personal:</strong> {res.personal?.name || '—'} |{' '}
                                  {res.personal?.email || '—'} | {res.personal?.phone || '—'}
                                </div>
                                <div>
                                  <strong>Counts:</strong> Experience ({res.experience_count || 0}),
                                  Education ({res.education_count || 0}), Skills ({res.skills_count || 0})
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!summary && !isLoading && (
        <div className="empty-benchmark-prompt">
          <Zap size={32} />
          <p>Select a benchmark suite and click "Run Benchmark" to evaluate.</p>
        </div>
      )}
    </div>
  );
};
