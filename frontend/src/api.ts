import type {
  BenchmarkSummary,
  ConfigResponse,
  DiagnosticResponse,
  FixtureItem,
  HealthResponse,
  ParseResponse,
} from './types';

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = 'Request failed';
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        if (typeof errJson.detail === 'string') {
          errorDetail = errJson.detail;
        } else if (errJson.detail.message) {
          errorDetail = errJson.detail.message;
          if (errJson.detail.violations && errJson.detail.violations.length > 0) {
            errorDetail += `\nViolations: ${errJson.detail.violations.join(', ')}`;
          }
        } else {
          errorDetail = JSON.stringify(errJson.detail);
        }
      } else if (errJson.message) {
        errorDetail = errJson.message;
      }
    } catch {
      errorDetail = `${response.status} ${response.statusText}`;
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/health`);
  return handleResponse<HealthResponse>(res);
}

export async function getConfig(): Promise<ConfigResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/config`);
  return handleResponse<ConfigResponse>(res);
}

export async function parseResume(file: File): Promise<ParseResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/api/v1/parse`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<ParseResponse>(res);
}

export async function parseDiagnostic(
  file: File,
  maxBlocks: number = 500,
  maxTextLength: number = 200
): Promise<DiagnosticResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const url = `${API_BASE_URL}/api/v1/parse/diagnostic?max_blocks=${maxBlocks}&max_text_length=${maxTextLength}`;
  const res = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<DiagnosticResponse>(res);
}

export async function getFixtures(): Promise<FixtureItem[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/fixtures`);
  return handleResponse<FixtureItem[]>(res);
}

export async function parseFixture(fixtureId: string): Promise<ParseResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/v1/fixtures/${encodeURIComponent(fixtureId)}/parse`,
    {
      method: 'POST',
    }
  );
  return handleResponse<ParseResponse>(res);
}

export async function runBenchmark(
  suite: string = 'regression_12',
  provider?: string,
  model?: string,
  representation?: string
): Promise<BenchmarkSummary> {
  const res = await fetch(`${API_BASE_URL}/api/v1/benchmark`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ suite, provider, model, representation }),
  });
  return handleResponse<BenchmarkSummary>(res);
}
