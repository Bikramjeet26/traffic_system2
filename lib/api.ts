// Hugging Face Spaces backend (override via NEXT_PUBLIC_API_BASE on Vercel)
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, '') ??
  'https://bikki26-traffic-management-system.hf.space'

// ── Types ───────────────────────────────────────────────────────────────────

export interface GpsCoords {
  latitude: number
  longitude: number
}

export interface PlateDetection {
  text: string
  confidence: number
  box: [number, number, number, number]
}

export interface ViolationDetection {
  type: 'no_helmet' | 'no_seatbelt' | string
  vehicle_class: string
  confidence: number
  box: [number, number, number, number]
}

export interface EvidenceRecord {
  timestamp: string
  gps: [number, number]
  plates: PlateDetection[]
  violations: ViolationDetection[]
  annotated_image_path: string
}

export interface AnalyzeResponse {
  success: boolean
  timestamp: string
  gps: GpsCoords
  violations_count: number
  plates_count: number
  evidence: EvidenceRecord
  annotated_image_base64?: string
  error?: string
}

export interface StatsResponse {
  total_records: number
  total_violations: number
  total_plates: number
  average_violations_per_record: number
  violation_counts: Record<string, number>
  vehicle_class_counts: Record<string, number>
}

export interface HealthResponse {
  status: string
  evidence_folder_exists: boolean
}

interface ApiEnvelope<T> {
  success?: boolean
  error?: string
  violations?: T
  stats?: T
}

async function parseApiError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { error?: string; detail?: string | { msg?: string }[] }
    if (typeof body.error === 'string') return body.error
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      return body.detail.map(d => d.msg ?? JSON.stringify(d)).join(', ')
    }
  } catch {
    // ignore JSON parse errors
  }
  return res.statusText || `Request failed (${res.status})`
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    throw new Error(await parseApiError(res))
  }
  return res.json() as Promise<T>
}

// ── Fetcher helpers ─────────────────────────────────────────────────────────

export async function fetchViolations(): Promise<EvidenceRecord[]> {
  const data = await apiFetch<ApiEnvelope<EvidenceRecord[]>>('/violations')
  return data.violations ?? []
}

export async function fetchStats(): Promise<StatsResponse> {
  const data = await apiFetch<ApiEnvelope<StatsResponse>>('/stats')
  if (!data.stats) {
    throw new Error('Stats response missing stats payload')
  }
  return data.stats
}

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health')
}

export async function analyzeImage(
  file: File,
  opts?: { timestamp?: string; gpsLat?: number; gpsLon?: number }
): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append('file', file)
  if (opts?.timestamp) form.append('timestamp', opts.timestamp)
  if (opts?.gpsLat !== undefined) form.append('gps_lat', String(opts.gpsLat))
  if (opts?.gpsLon !== undefined) form.append('gps_lon', String(opts.gpsLon))

  return apiFetch<AnalyzeResponse>('/analyze', { method: 'POST', body: form })
}

/**
 * Build the URL for an annotated evidence image served from the /evidence static route.
 * annotated_image_path is typically "evidence/2025-06-18T14-30-22.123Z.jpg"
 */
export function evidenceImageUrl(annotatedImagePath: string): string {
  const filename = annotatedImagePath.replace(/^evidence[/\\]/, '')
  return `${API_BASE}/evidence/${encodeURIComponent(filename)}`
}

/**
 * Build URL for a JSON evidence file by its timestamp string.
 */
export function evidenceJsonUrl(timestamp: string): string {
  const safe = timestamp.replace(/:/g, '-')
  return `${API_BASE}/evidence/${encodeURIComponent(safe)}.json`
}

// SWR key factories (prevents magic strings across components)
export const SWR_KEYS = {
  violations: '/violations',
  stats: '/stats',
  health: '/health',
} as const
