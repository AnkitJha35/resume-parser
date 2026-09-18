export interface HealthResponse {
  status: string;
  parser_version: string;
  application_version: string;
}

export interface ConfigResponse {
  provider: string;
  model: string;
  representation: string;
  parser_version: string;
  ocr_available: boolean;
  ocr_engine?: string | null;
  two_pass: boolean;
  fallback_enabled: boolean;
}

export interface ParseMetadata {
  filename: string;
  pageCount: number;
  ocrUsed: boolean;
  provider: string;
  model: string;
  representation: string;
  latencyMs: number;
  requestCount: number;
}

export interface ProvenanceRecord {
  canonicalField: string;
  extractedValue?: any;
  rawValue?: string | null;
  sourceBlockIds: string[];
  sourceTexts: string[];
  pageNumbers: number[];
  confidence?: number | null;
}

export interface PersonalInfo {
  name?: string | null;
  email?: string | null;
  phone?: string | null;
  location?: string | null;
  linkedin?: string | null;
  github?: string | null;
  portfolio?: string | null;
}

export interface ExperienceItem {
  company?: string | null;
  designation?: string | null;
  location?: string | null;
  startDate?: string | null;
  endDate?: string | null;
  current?: boolean | null;
  description?: string | null;
  skills?: string[] | null;
  confidence?: number | null;
}

export interface EducationItem {
  institution?: string | null;
  degree?: string | null;
  fieldOfStudy?: string | null;
  startDate?: string | null;
  endDate?: string | null;
  grade?: string | null;
  confidence?: number | null;
}

export interface ProjectItem {
  name?: string | null;
  description?: string | null;
  technologies?: string[] | null;
  startDate?: string | null;
  endDate?: string | null;
  current?: boolean | null;
  url?: string | null;
  confidence?: number | null;
}

export interface CertificationItem {
  name?: string | null;
  issuingOrganization?: string | null;
  issueDate?: string | null;
  expiryDate?: string | null;
  credentialId?: string | null;
  credentialUrl?: string | null;
  confidence?: number | null;
}

export interface Resume {
  schemaVersion?: string;
  parserVersion: string;
  personal: PersonalInfo;
  summary?: string | null;
  skills: string[];
  experience: ExperienceItem[];
  education: EducationItem[];
  projects: ProjectItem[];
  certifications: CertificationItem[];
  achievements: string[];
  languages: string[];
  metadata?: Record<string, any>;
}

export type ParseStatus =
  | 'SUCCESS'
  | 'PARTIAL'
  | 'VALIDATION_FAILED'
  | 'EXTRACTION_FAILED'
  | 'ERROR'
  | 'success'
  | 'validation_failed'
  | string;

export interface ParseResponse {
  success: boolean;
  status: ParseStatus;
  resume: Resume | null;
  violations?: string[];
  metadata: ParseMetadata;
  diagnostics: {
    provenance?: ProvenanceRecord[];
    archetype?: string;
    violations?: string[];
    blockClassifications?: Array<{ block_id: string; category: string }>;
    totalBlocks?: number;
    tokenUsage?: {
      promptTokens?: number | null;
      outputTokens?: number | null;
      totalTokens?: number | null;
    };
    [key: string]: any;
  };
}

export interface DiagnosticBlockItem {
  block_id: string;
  page_number: number;
  reading_order: number;
  text: string;
  bbox: [number, number, number, number];
  suggested_role?: string | null;
  table_id?: string | null;
  row_index?: number | null;
  column_index?: number | null;
  is_continuation?: boolean;
  section_hint?: string | null;
}

export interface DiagnosticPageItem {
  page_number: number;
  width: number;
  height: number;
  region_count: number;
}

export interface DiagnosticRegionItem {
  page_number: number;
  region_id: string;
  bbox: [number, number, number, number];
  reading_order: number;
  region_type: string;
  line_count: number;
}

export interface DiagnosticTableItem {
  table_id: string;
  page_number: number;
  row_count: number;
  column_count: number;
  headers: string[];
}

export interface DiagnosticResponse {
  page_count: number;
  block_count: number;
  archetype: string;
  pages: DiagnosticPageItem[];
  regions: DiagnosticRegionItem[];
  blocks: DiagnosticBlockItem[];
  tables: DiagnosticTableItem[];
  truncated: boolean;
}

export interface FixtureItem {
  id: string;
  suite: string;
  filename: string;
  candidate_name?: string | null;
  target_domain?: string | null;
  archetype?: string | null;
  page_count_estimate?: number | null;
  has_tables?: boolean | null;
  notes?: string | null;
}

export interface SemanticParseResult {
  filename: string;
  archetype: string;
  target_domain: string;
  semantic_success: boolean;
  status: string;
  failure_type?: string | null;
  error_message?: string | null;
  validation_violations: string[];
  passed_validation: boolean;
  elapsed_seconds: number;
  provider: string;
  model: string;
  representation: string;
  hard_correctness_passed: boolean;
  hard_correctness_violations: string[];
  entity_completeness_pct: number;
  field_completeness_pct: number;
  skills_metrics?: Record<string, any>;
  personal?: Record<string, any>;
  skills_count?: number;
  experience_count?: number;
  education_count?: number;
  diagnostics?: string[];
  notes?: string;
  usage?: Record<string, any> | null;
}

export interface BenchmarkSummary {
  timestamp: string;
  suite_id: string;
  provider: string;
  model: string;
  representation: string;
  total_cases: number;
  successful_cases: number;
  extraction_failures: number;
  validation_failures: number;
  completeness_failures: number;
  parser_exceptions: number;
  hard_correctness_pass_rate_pct: number;
  entity_completeness_rate_pct: number;
  field_completeness_rate_pct: number;
  avg_skills_recall_pct?: number | null;
  avg_requests_per_resume: number;
  total_elapsed_seconds: number;
  total_tokens?: number | null;
  results: SemanticParseResult[];
}
