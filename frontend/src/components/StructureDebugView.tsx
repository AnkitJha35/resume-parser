import React, { useState, useMemo } from 'react';
import type { DiagnosticResponse } from '../types';
import { Search, Filter, Layers, Table, FileText, AlertCircle } from 'lucide-react';

interface StructureDebugViewProps {
  diagnostic: DiagnosticResponse;
}

export const StructureDebugView: React.FC<StructureDebugViewProps> = ({
  diagnostic,
}) => {
  const [selectedPage, setSelectedPage] = useState<string>('all');
  const [selectedRole, setSelectedRole] = useState<string>('all');
  const [selectedTable, setSelectedTable] = useState<string>('all');
  const [selectedColumn, setSelectedColumn] = useState<string>('all');
  const [searchText, setSearchText] = useState<string>('');

  // Extract distinct roles, tables, columns for filters
  const roles = useMemo(() => {
    const set = new Set<string>();
    diagnostic.blocks.forEach((b) => {
      if (b.suggested_role) set.add(b.suggested_role);
    });
    return Array.from(set).sort();
  }, [diagnostic.blocks]);

  const tables = useMemo(() => {
    const set = new Set<string>();
    diagnostic.blocks.forEach((b) => {
      if (b.table_id) set.add(b.table_id);
    });
    return Array.from(set).sort();
  }, [diagnostic.blocks]);

  const columns = useMemo(() => {
    const set = new Set<number>();
    diagnostic.blocks.forEach((b) => {
      if (b.column_index !== null && b.column_index !== undefined) {
        set.add(b.column_index);
      }
    });
    return Array.from(set).sort((a, b) => a - b);
  }, [diagnostic.blocks]);

  const filteredBlocks = useMemo(() => {
    return diagnostic.blocks.filter((b) => {
      if (selectedPage !== 'all' && String(b.page_number) !== selectedPage) {
        return false;
      }
      if (selectedRole !== 'all' && b.suggested_role !== selectedRole) {
        return false;
      }
      if (selectedTable !== 'all') {
        if (selectedTable === 'only_tables' && !b.table_id) return false;
        if (selectedTable === 'no_tables' && b.table_id) return false;
        if (selectedTable !== 'only_tables' && selectedTable !== 'no_tables' && b.table_id !== selectedTable) {
          return false;
        }
      }
      if (selectedColumn !== 'all' && String(b.column_index) !== selectedColumn) {
        return false;
      }
      if (searchText) {
        const text = searchText.toLowerCase();
        const matchesText = b.text.toLowerCase().includes(text);
        const matchesId = b.block_id.toLowerCase().includes(text);
        if (!matchesText && !matchesId) return false;
      }
      return true;
    });
  }, [diagnostic.blocks, selectedPage, selectedRole, selectedTable, selectedColumn, searchText]);

  return (
    <div className="structure-viewer card">
      <div className="structure-header">
        <div>
          <h3>Structural & Layout Debug View</h3>
          <p className="structure-subtitle">
            Inspect Document IR blocks, visual reading orders, detected table grid bindings, and structural roles.
          </p>
        </div>

        <div className="structure-metrics">
          <span className="badge badge-meta">
            <FileText size={13} /> Pages: <strong>{diagnostic.page_count}</strong>
          </span>
          <span className="badge badge-meta">
            <Layers size={13} /> Regions: <strong>{diagnostic.regions.length}</strong>
          </span>
          <span className="badge badge-meta">
            <Table size={13} /> Tables: <strong>{diagnostic.tables.length}</strong>
          </span>
          <span className="badge badge-meta">
            Total Blocks: <strong>{diagnostic.block_count}</strong>
          </span>
          {diagnostic.truncated && (
            <span className="badge badge-warning">
              <AlertCircle size={13} /> Sliced to {diagnostic.blocks.length} blocks
            </span>
          )}
        </div>
      </div>

      {/* FILTERS */}
      <div className="filters-container">
        <div className="filter-row">
          <div className="filter-group">
            <label>
              <Filter size={12} /> Page:
            </label>
            <select
              className="select-input select-sm"
              value={selectedPage}
              onChange={(e) => setSelectedPage(e.target.value)}
            >
              <option value="all">All Pages</option>
              {diagnostic.pages.map((p) => (
                <option key={p.page_number} value={String(p.page_number)}>
                  Page {p.page_number}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Role:</label>
            <select
              className="select-input select-sm"
              value={selectedRole}
              onChange={(e) => setSelectedRole(e.target.value)}
            >
              <option value="all">All Roles</option>
              {roles.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Table Filter:</label>
            <select
              className="select-input select-sm"
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
            >
              <option value="all">All Blocks</option>
              <option value="only_tables">In Any Table</option>
              <option value="no_tables">Non-Table Only</option>
              {tables.map((t) => (
                <option key={t} value={t}>
                  Table: {t}
                </option>
              ))}
            </select>
          </div>

          {columns.length > 0 && (
            <div className="filter-group">
              <label>Column Index:</label>
              <select
                className="select-input select-sm"
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(e.target.value)}
              >
                <option value="all">All Columns</option>
                {columns.map((c) => (
                  <option key={c} value={String(c)}>
                    Col {c}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="filter-group search-filter-group">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              className="search-input search-sm"
              placeholder="Search block text / ID..."
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
            />
          </div>
        </div>

        <div className="filter-summary">
          Showing <strong>{filteredBlocks.length}</strong> of{' '}
          <strong>{diagnostic.blocks.length}</strong> loaded blocks
        </div>
      </div>

      {/* TABLE */}
      <div className="table-responsive">
        <table className="debug-table">
          <thead>
            <tr>
              <th style={{ width: '6%' }}>Pg</th>
              <th style={{ width: '8%' }}>Order</th>
              <th style={{ width: '14%' }}>Role</th>
              <th style={{ width: '12%' }}>Table / Col</th>
              <th style={{ width: '12%' }}>Block ID</th>
              <th style={{ width: '16%' }}>BBox (x0, y0, x1, y1)</th>
              <th style={{ width: '32%' }}>Block Text</th>
            </tr>
          </thead>
          <tbody>
            {filteredBlocks.map((b) => (
              <tr key={b.block_id} className={b.table_id ? 'row-table-block' : ''}>
                <td>
                  <span className="badge badge-page">P{b.page_number}</span>
                </td>
                <td>
                  <span className="order-num">#{b.reading_order}</span>
                </td>
                <td>
                  {b.suggested_role ? (
                    <span className={`badge badge-role role-${b.suggested_role.toLowerCase()}`}>
                      {b.suggested_role}
                    </span>
                  ) : (
                    <span className="text-muted">—</span>
                  )}
                </td>
                <td>
                  {b.table_id ? (
                    <div className="table-binding-badge">
                      <span className="tbl-id">{b.table_id}</span>
                      <span className="tbl-coords">
                        R{b.row_index !== null ? b.row_index : '?'}:C
                        {b.column_index !== null ? b.column_index : '?'}
                      </span>
                    </div>
                  ) : (
                    <span className="text-muted">—</span>
                  )}
                </td>
                <td>
                  <code className="code-id">{b.block_id}</code>
                </td>
                <td>
                  <span className="code-bbox">
                    [{b.bbox.join(', ')}]
                  </span>
                </td>
                <td>
                  <div className="block-text-cell">{b.text}</div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
