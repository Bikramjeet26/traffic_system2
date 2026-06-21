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

export type ViolationType =
  | 'no_helmet'
  | 'no_seatbelt'
  | 'triple_riding'
  | 'red_light'
  | 'stop_line'
  | 'red_light_violation'
  | 'stop_line_violation'
  | (string & {})

export interface ViolationDetection {
  type: ViolationType
  vehicle_class: string
  confidence: number
  box: [number, number, number, number]
  /** Present for triple_riding violations */
  person_count?: number
}

/** Human-readable label for any violation type (shared across pages). */
export function violationLabel(type: string): string {
  switch (type) {
    case 'no_helmet':
      return 'No Helmet'
    case 'no_seatbelt':
      return 'No Seatbelt'
    case 'triple_riding':
      return 'Triple Riding'
    case 'red_light':
    case 'red_light_violation':
      return 'Red Light'
    case 'stop_line':
    case 'stop_line_violation':
      return 'Stop Line'
    default:
      return type
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
  }
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

// ── Face Match (criminal detection) ──────────────────────────────────────────

export interface FaceMatchEntry {
  similarity: number
  distance: number
  is_match?: boolean
  face_index?: number | null
  box?: [number, number, number, number] | null
}

export interface FaceCompareResult {
  success: boolean
  mode: 'image_to_image'
  is_match: boolean
  threshold?: number
  faces_in_person?: number
  faces_in_target?: number
  best_match?: FaceMatchEntry | null
  all_matches?: FaceMatchEntry[]
  error?: string
}

export interface ViolationSummary {
  timestamp: string
  gps: { latitude: number | null; longitude: number | null }
  violations_count: number
  plates_count: number
  annotated_image_path?: string | null
}

export interface FaceScanMatch {
  violation: ViolationSummary
  is_match: boolean
  similarity: number
  distance: number
  face_index?: number | null
  box?: [number, number, number, number] | null
  faces_in_violation?: number
}

export interface FaceScanResult {
  success: boolean
  mode: 'violation_database_scan' | 'violation_record'
  is_match: boolean
  threshold?: number
  total_violations_scanned?: number
  total_violations_skipped?: number
  matches_found?: number
  best_match?: FaceScanMatch | FaceMatchEntry | null
  all_results?: FaceScanMatch[]
  positive_matches?: FaceScanMatch[]
  violation?: ViolationSummary
  faces_in_target?: number
  error?: string
}

/** Mode 1: compare a reference person image against a target image. */
export async function faceMatchCompare(
  personImage: File,
  targetImage: File
): Promise<FaceCompareResult> {
  const form = new FormData()
  form.append('person_image', personImage)
  form.append('target_image', targetImage)
  return apiFetch<FaceCompareResult>('/face-match/compare', { method: 'POST', body: form })
}

/**
 * Mode 2/3: compare a reference person against a stored violation by id,
 * or scan the entire violation database for a match.
 */
export async function faceMatchViolation(
  personImage: File,
  opts: { violationId?: string; scanAll?: boolean }
): Promise<FaceScanResult> {
  const form = new FormData()
  form.append('person_image', personImage)
  if (opts.scanAll) {
    form.append('scan_all_violations', 'true')
  } else if (opts.violationId) {
    form.append('violation_id', opts.violationId)
  }
  return apiFetch<FaceScanResult>('/face-match/violation', { method: 'POST', body: form })
}

// SWR key factories (prevents magic strings across components)
export const SWR_KEYS = {
  violations: '/violations',
  stats: '/stats',
  health: '/health',
} as const
