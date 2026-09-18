import React, { useState } from 'react';
import type { ProvenanceRecord } from '../types';
import { Search, ShieldCheck, FileText, Hash, Layers } from 'lucide-react';

interface ProvenanceViewProps {
  records: ProvenanceRecord[];
  archetype?: string;
  totalBlocks?: number;
}

export const ProvenanceView: React.FC<ProvenanceViewProps> = ({
  records,
  archetype,
  totalBlocks,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedField, setSelectedField] = useState<string>('all');

  const fieldCategories = Array.from(
    new Set(records.map((r) => r.canonicalField.split('.')[0].split('[')[0]))
  );

  const filteredRecords = records.filter((rec) => {
    const matchesCategory =
      selectedField === 'all' || rec.canonicalField.startsWith(selectedField);
    const term = searchTerm.toLowerCase();
    const matchesSearch =
      !term ||
      rec.canonicalField.toLowerCase().includes(term) ||
      String(rec.extractedValue || '').toLowerCase().includes(term) ||
      rec.sourceTexts.some((txt) => txt.toLowerCase().includes(term)) ||
      rec.sourceBlockIds.some((id) => id.toLowerCase().includes(term));
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="provenance-viewer card">
      <div className="provenance-header">
        <div>
          <h3>Field Provenance & Grounded Evidence</h3>
          <p className="provenance-subtitle">
            Traces canonical fields back to OCR/Document IR source blocks and page locations.
          </p>
        </div>

        <div className="provenance-meta-badges">
          {archetype && (
            <span className="badge badge-meta">
              <Layers size={13} /> Archetype: <strong>{archetype}</strong>
            </span>
          )}
          {totalBlocks !== undefined && (
            <span className="badge badge-meta">
              <Hash size={13} /> Document Blocks: <strong>{totalBlocks}</strong>
            </span>
          )}
          <span className="badge badge-meta">
            <ShieldCheck size={13} /> Grounded Fields: <strong>{records.length}</strong>
          </span>
        </div>
      </div>

      <div className="filter-bar">
        <div className="search-input-wrap">
          <Search size={15} className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Search fields, values, source text, block IDs..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="category-chips">
          <button
            className={`chip ${selectedField === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedField('all')}
          >
            All Fields ({records.length})
          </button>
          {fieldCategories.map((cat) => (
            <button
              key={cat}
              className={`chip ${selectedField === cat ? 'active' : ''}`}
              onClick={() => setSelectedField(cat)}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {filteredRecords.length > 0 ? (
        <div className="table-responsive">
          <table className="provenance-table">
            <thead>
              <tr>
                <th style={{ width: '22%' }}>Canonical Output</th>
                <th style={{ width: '30%' }}>Extracted Value</th>
                <th style={{ width: '12%' }}>Source Evidence</th>
                <th style={{ width: '36%' }}>Verbatim Source Snippets</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.map((rec, idx) => (
                <tr key={idx}>
                  <td>
                    <div className="field-tag">
                      <span className="badge badge-field">{rec.canonicalField}</span>
                    </div>
                  </td>
                  <td>
                    <div className="val-box">
                      <span className="canonical-val">
                        {typeof rec.extractedValue === 'boolean'
                          ? rec.extractedValue ? 'true' : 'false'
                          : String(rec.extractedValue || '—')}
                      </span>
                      {rec.rawValue && rec.rawValue !== rec.extractedValue && (
                        <span className="raw-val" title="Verbatim Raw Text from Source">
                          raw: "{rec.rawValue}"
                        </span>
                      )}
                    </div>
                  </td>
                  <td>
                    <div className="evidence-refs">
                      <div className="pages-row">
                        <FileText size={12} />
                        <span>
                          {rec.pageNumbers && rec.pageNumbers.length > 0
                            ? `Page ${rec.pageNumbers.join(', ')}`
                            : 'Page 1'}
                        </span>
                      </div>
                      <div className="block-ids">
                        {rec.sourceBlockIds.map((bId) => (
                          <span key={bId} className="badge badge-block-id" title={bId}>
                            {bId}
                          </span>
                        ))}
                      </div>
                    </div>
                  </td>
                  <td>
                    <div className="snippets-column">
                      {rec.sourceTexts && rec.sourceTexts.length > 0 ? (
                        rec.sourceTexts.map((txt, sIdx) => (
                          <div key={sIdx} className="snippet-item">
                            "{txt}"
                          </div>
                        ))
                      ) : (
                        <span className="empty-snippet">—</span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="empty-state">No provenance records match your search criteria.</p>
      )}
    </div>
  );
};
