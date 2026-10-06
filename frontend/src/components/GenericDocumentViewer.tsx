import React, { useState, useEffect, useMemo, useRef } from 'react';
import type { DocumentBlock, DocumentSection, DocumentStructure } from '../types';
import { Copy, Check, ChevronRight, FileText, ListTree, Menu } from 'lucide-react';
import './GenericDocumentViewer.css';

interface GenericDocumentViewerProps {
  documentStructure: DocumentStructure;
}

interface OutlineItem {
  id: string;
  label: string;
  level: number;
  isUnheaded: boolean;
}

export const GenericDocumentViewer: React.FC<GenericDocumentViewerProps> = ({ documentStructure }) => {
  const [copied, setCopied] = useState(false);
  const [activeSectionId, setActiveSectionId] = useState<string>('');
  const [isMobileOutlineOpen, setIsMobileOutlineOpen] = useState(false);
  const readingCanvasRef = useRef<HTMLDivElement>(null);

  const sections = documentStructure.sections || [];
  const pageCount = documentStructure.page_count || 1;

  // Build dynamic outline tree entirely from document_structure using useMemo
  const outlineItems: OutlineItem[] = useMemo(() => {
    const items: OutlineItem[] = [];
    const secList = documentStructure.sections || [];
    secList.forEach((sec, idx) => {
      const isUnheaded = !sec.heading || sec.heading.trim() === '';
      const secId = isUnheaded ? 'gdv-sec-unheaded' : `gdv-sec-${idx}`;
      const label = isUnheaded ? 'Overview' : sec.heading || '';

      items.push({
        id: secId,
        label,
        level: sec.level || 1,
        isUnheaded,
      });

      if (sec.subsections && sec.subsections.length > 0) {
        sec.subsections.forEach((sub, subIdx) => {
          const subId = `gdv-sec-${idx}-${subIdx}`;
          items.push({
            id: subId,
            label: sub.heading || `Subsection ${subIdx + 1}`,
            level: sub.level || 2,
            isUnheaded: false,
          });
        });
      }
    });
    return items;
  }, [documentStructure.sections]);

  const currentActiveId = activeSectionId || (outlineItems.length > 0 ? outlineItems[0].id : '');

  // Track active section via IntersectionObserver on reading canvas
  useEffect(() => {
    const canvas = readingCanvasRef.current;
    if (!canvas || typeof IntersectionObserver === 'undefined') return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSectionId(entry.target.id);
            break;
          }
        }
      },
      {
        root: canvas,
        rootMargin: '-10% 0px -70% 0px',
        threshold: 0.1,
      }
    );

    const sectionElements = canvas.querySelectorAll('.gdv-section, .gdv-section-unheaded');
    sectionElements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [documentStructure]);

  const handleScrollTo = (id: string) => {
    setActiveSectionId(id);
    setIsMobileOutlineOpen(false);
    const targetEl = document.getElementById(id);
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(documentStructure, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="gdv-root">
      {/* TOP TOOLBAR */}
      <header className="gdv-toolbar">
        <div className="gdv-toolbar-left">
          <div className="gdv-title-badge">
            <div className="gdv-title-icon">
              <FileText size={16} />
            </div>
            <span className="gdv-title-text">Document Structure</span>
          </div>

          <span className="gdv-meta-pill">
            <span className="gdv-meta-dot" />
            {sections.length} {sections.length === 1 ? 'section' : 'sections'} · {pageCount} {pageCount === 1 ? 'page' : 'pages'}
          </span>

          <button
            type="button"
            className="gdv-outline-toggle-btn"
            onClick={() => setIsMobileOutlineOpen(!isMobileOutlineOpen)}
            aria-label="Toggle document outline"
          >
            <Menu size={14} />
            <span>Outline</span>
          </button>
        </div>

        <div className="gdv-toolbar-right">
          <button
            type="button"
            onClick={handleCopyJson}
            className={`gdv-btn-copy ${copied ? 'copied' : ''}`}
            title="Copy exact canonical JSON representation"
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
            <span>{copied ? 'Copied' : 'Copy JSON'}</span>
          </button>
        </div>
      </header>

      {/* TWO-PANE LAYOUT */}
      <div className="gdv-layout">
        {/* LEFT PANE: DYNAMIC DOCUMENT OUTLINE */}
        <aside className={`gdv-sidebar ${isMobileOutlineOpen ? 'mobile-open' : 'collapsed'}`}>
          <div className="gdv-sidebar-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <ListTree size={14} />
              <span>Document Outline</span>
            </span>
            <span style={{ fontSize: '10px', color: '#94a3b8' }}>{outlineItems.length}</span>
          </div>

          <nav className="gdv-outline-nav" aria-label="Document Outline Navigation">
            {outlineItems.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`gdv-outline-item ${item.level === 2 ? 'level-2' : 'level-1'} ${
                  currentActiveId === item.id ? 'active' : ''
                }`}
                onClick={() => handleScrollTo(item.id)}
                title={item.label}
              >
                <span className="gdv-outline-bullet" />
                <span className="gdv-outline-label">{item.label}</span>
              </button>
            ))}
          </nav>
        </aside>

        {/* RIGHT PANE: MAIN DOCUMENT READING SURFACE */}
        <main className="gdv-reading-canvas" ref={readingCanvasRef}>
          <article className="gdv-paper">
            {sections.length > 0 ? (
              sections.map((sec, idx) => (
                <SectionView
                  key={`sec-${idx}`}
                  section={sec}
                  sectionIndex={idx}
                  depth={0}
                />
              ))
            ) : (
              <div className="gdv-empty-state">
                <FileText size={36} color="#94a3b8" />
                <span className="gdv-empty-title">No structured sections detected</span>
                <p style={{ margin: 0, fontSize: '13px' }}>
                  Upload a PDF document to view its generic structural hierarchy.
                </p>
              </div>
            )}
          </article>
        </main>
      </div>
    </div>
  );
};

interface SectionViewProps {
  section: DocumentSection;
  sectionIndex: number;
  depth: number;
}

const SectionView: React.FC<SectionViewProps> = ({ section, sectionIndex, depth }) => {
  const isUnheaded = !section.heading || section.heading.trim() === '';
  const level = section.level || 1;
  const sectionDomId = isUnheaded
    ? 'gdv-sec-unheaded'
    : depth === 0
    ? `gdv-sec-${sectionIndex}`
    : `gdv-sec-${sectionIndex}-${depth}`;

  const groupedBlocks = groupBlocks(section.blocks || []);

  return (
    <section
      id={sectionDomId}
      className={isUnheaded ? 'gdv-section-unheaded' : 'gdv-section'}
      data-level={level}
    >
      {/* SECTION HEADING (ONLY WHEN PRESENT) */}
      {!isUnheaded && (
        <header className="gdv-section-header">
          {level === 1 ? (
            <h2 className="gdv-section-title-l1">{section.heading}</h2>
          ) : (
            <h3 className="gdv-section-title-l2">
              <ChevronRight size={14} color="#64748b" />
              <span>{section.heading}</span>
            </h3>
          )}
        </header>
      )}

      {/* SECTION CONTENT BLOCKS */}
      <div className="gdv-blocks-flow">
        {groupedBlocks.map((group, gIdx) => {
          if (group.type === 'list') {
            return (
              <ul key={`list-${gIdx}`} className="gdv-list">
                {group.blocks.map((b, bIdx) => (
                  <li key={`li-${bIdx}`} className="gdv-list-item">
                    {b.text}
                  </li>
                ))}
              </ul>
            );
          }

          if (group.type === 'table') {
            const tableBlock = group.blocks[0];
            const tableData = tableBlock.table_data;
            const geom = tableData?.visual_geometry;

            // Visual geometry configuration with fallback to true (standard grid)
            const hasOuterBorder = geom ? geom.has_outer_border !== false : true;
            const hasHorizBorders = geom ? geom.has_horizontal_borders !== false : true;
            const hasVertBorders = geom ? geom.has_vertical_borders !== false : true;
            const rowHBorders = geom?.horizontal_borders;
            const colVBorders = geom?.vertical_borders;

            // If table has 2D spatial form layout, render as generic geometry-first form container
            if (tableData?.is_form_layout) {
              const blocks =
                tableData.spatial_blocks && tableData.spatial_blocks.length > 0
                  ? tableData.spatial_blocks
                  : (tableData.spatial_rows || []).flatMap((r) => r.fields);

              if (blocks.length > 0) {
                // Calculate container vertical percentage from aspect ratio (ar = width / height => height / width = 1 / ar)
                const ar = tableData.aspect_ratio && tableData.aspect_ratio > 0 ? tableData.aspect_ratio : 4.0;
                const paddingBottomPct = Math.round((100 / ar) * 100) / 100;

                return (
                  <div
                    key={`tbl-${gIdx}`}
                    className={`gdv-table-container gdv-form-container ${!hasOuterBorder ? 'gdv-table-no-outer-border' : ''}`}
                  >
                    <div
                      className="gdv-form-canvas"
                      style={{
                        paddingBottom: `${paddingBottomPct}%`,
                      }}
                    >
                      {blocks.map((b, bIdx) => (
                        <div
                          key={`sblk-${bIdx}`}
                          className="gdv-form-block"
                          style={{
                            left: `${b.relative_x}%`,
                            top: `${b.relative_y}%`,
                            width: `${Math.max(b.relative_width, 8)}%`,
                            minHeight: b.relative_height ? `${b.relative_height}%` : undefined,
                          }}
                        >
                          <span className="gdv-form-text">{b.text}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              }
            }

            const headers = tableData?.headers || [];
            const rows = tableData?.rows || [];
            const columnWidths = tableData?.column_widths;
            const hasWidths = columnWidths && columnWidths.length > 0;

            return (
              <div
                key={`tbl-${gIdx}`}
                className={`gdv-table-container ${!hasOuterBorder ? 'gdv-table-no-outer-border' : ''}`}
              >
                <table className={`gdv-table ${hasWidths ? 'gdv-table-fixed' : ''}`}>
                  {hasWidths && (
                    <colgroup>
                      {columnWidths.map((w, wIdx) => (
                        <col key={`col-${wIdx}`} style={{ width: `${w}%` }} />
                      ))}
                    </colgroup>
                  )}
                  {headers.length > 0 && (
                    <thead style={!hasHorizBorders ? { borderBottom: 'none' } : undefined}>
                      <tr>
                        {headers.map((h, hIdx) => {
                          const showColBorder = hasVertBorders && (!colVBorders || colVBorders[hIdx] !== false);
                          return (
                            <th
                              key={`th-${hIdx}`}
                              style={!showColBorder ? { borderRight: 'none' } : undefined}
                            >
                              {h ? h : '\u00A0'}
                            </th>
                          );
                        })}
                      </tr>
                    </thead>
                  )}
                  <tbody>
                    {rows.map((row, rIdx) => {
                      const showRowBorder = hasHorizBorders && (!rowHBorders || rowHBorders[rIdx] !== false);
                      return (
                        <tr key={`tr-${rIdx}`}>
                          {row.map((cell, cIdx) => {
                            const showColBorder = hasVertBorders && (!colVBorders || colVBorders[cIdx] !== false);
                            return (
                              <td
                                key={`td-${cIdx}`}
                                style={{
                                  ...(!showColBorder ? { borderRight: 'none' } : {}),
                                  ...(!showRowBorder ? { borderBottom: 'none' } : {}),
                                }}
                              >
                                {cell ? cell : '\u00A0'}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            );
          }

          // Paragraph / standard text blocks
          return group.blocks.map((b, bIdx) => (
            <p key={`p-${gIdx}-${bIdx}`} className="gdv-paragraph">
              {b.text}
            </p>
          ));
        })}
      </div>

      {/* NESTED SUBSECTIONS */}
      {section.subsections && section.subsections.length > 0 && (
        <div className="gdv-subsections">
          {section.subsections.map((sub, sIdx) => (
            <SectionView
              key={`sub-${sIdx}`}
              section={sub}
              sectionIndex={sIdx}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </section>
  );
};

interface BlockGroup {
  type: 'list' | 'table' | 'paragraph';
  blocks: DocumentBlock[];
}

function groupBlocks(blocks: DocumentBlock[]): BlockGroup[] {
  const groups: BlockGroup[] = [];
  let currentList: DocumentBlock[] = [];

  const flushList = () => {
    if (currentList.length > 0) {
      groups.push({ type: 'list', blocks: [...currentList] });
      currentList = [];
    }
  };

  for (const b of blocks) {
    if (b.type === 'list_item') {
      currentList.push(b);
    } else if (b.type === 'table' || (b.table_data && (b.table_data.rows?.length || 0) > 0)) {
      flushList();
      groups.push({ type: 'table', blocks: [b] });
    } else {
      flushList();
      groups.push({ type: 'paragraph', blocks: [b] });
    }
  }

  flushList();
  return groups;
}
