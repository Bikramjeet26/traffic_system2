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

export interface StopLineViolation {
  track_id: number
  frame_idx: number
  bbox: [number, number, number, number]
  prev_point?: [number, number]
  curr_point?: [number, number]
  light_state: string
  type: 'stop_line_violation' | 'red_light_violation'
}

export interface AnalyzeVideoResponse {
  success: boolean
  annotated_video_url: string
  annotated_video_path?: string
  results_json_url: string
  results_json_path?: string
  violations_count: number
  violations: StopLineViolation[]
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

export interface HeatmapPoint {
  id: string
  lat: number
  lon: number
  timestamp: string
  weight: number
  violation_types: string[]
  violation_count: number
  image_url: string
}

export interface HeatmapResponse {
  success: boolean
  static_prefix: string
  total_points: number
  points: HeatmapPoint[]
}

export interface CollectionViolationRecord {
  id: string
  timestamp: string
  gps_lat: number
  gps_lon: number
  gps: [number, number]
  plates: PlateDetection[]
  violations: ViolationDetection[]
  image_filename: string
  json_filename: string
  image_url: string
  json_url: string
  system_outputs: {
    vehicle_detected: boolean
    license_plate_detected: boolean
    ocr_detected: boolean
    violation_types: string[]
    plate_texts: string[]
  }
}

export interface CollectionViolationsResponse {
  success: boolean
  evidence_folder: string
  static_prefix: string
  total_records: number
  records: CollectionViolationRecord[]
}

export interface ModelVerificationLabel {
  detected: boolean
  correct: boolean
  notes?: string
}

export interface VerifyViolationRequest {
  violation_id: string
  violation_confirmed: boolean
  ocr: ModelVerificationLabel
  license_plate: ModelVerificationLabel
  vehicle: ModelVerificationLabel
  helmet?: ModelVerificationLabel
  seatbelt?: ModelVerificationLabel
  annotation_notes?: string
}

export interface TrainingRecord {
  id: string
  source_violation_id: string
  verified_at: string
  category: string
  human_labels: Record<string, ModelVerificationLabel>
  annotation_notes?: string
  image_filename: string
  json_filename: string
  image_url: string
  json_url: string
}

export interface TrainingStats {
  total_confirmed: number
  total_corrections: number
  total_training_records: number
  confirmed_violation_types: Record<string, number>
  corrections_violation_types: Record<string, number>
}

export interface VerifyViolationResponse {
  success: boolean
  training_record: TrainingRecord
  training_stats: TrainingStats
}

export interface CollectionViolationDetailResponse {
  success: boolean
  record: CollectionViolationRecord
  system_outputs: {
    vehicle_detected: boolean
    license_plate_detected: boolean
    ocr_detected: boolean
    violation_types: string[]
    plate_texts: string[]
  }
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

export async function analyzeVideo(
  file: File,
  opts: {
    stopLine: [number, number, number, number]
    redLightLine: [number, number, number, number]
    initialLightState?: 'red' | 'green'
    confThres?: number
  }
): Promise<AnalyzeVideoResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('stop_line', JSON.stringify(opts.stopLine))
  form.append('red_light_line', JSON.stringify(opts.redLightLine))
  form.append('initial_light_state', opts.initialLightState ?? 'red')
  form.append('conf_thres', String(opts.confThres ?? 0.3))

  return apiFetch<AnalyzeVideoResponse>('/analyze-video', { method: 'POST', body: form })
}

/** Resolve a backend-relative evidence URL (e.g. /evidence/foo.mp4) to a full URL. */
export function evidenceAssetUrl(relativeUrl: string): string {
  if (relativeUrl.startsWith('http://') || relativeUrl.startsWith('https://')) {
    return relativeUrl
  }
  const path = relativeUrl.startsWith('/') ? relativeUrl : `/${relativeUrl}`
  return `${API_BASE}${path}`
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

// ── Collection & Heatmap Endpoints ──────────────────────────────────────────

export async function fetchCollectionViolations(
  includeZeroGps: boolean = true
): Promise<CollectionViolationRecord[]> {
  const data = await apiFetch<CollectionViolationsResponse>(
    `/collection/violations?include_zero_gps=${includeZeroGps}`
  )
  return data.records ?? []
}

export async function fetchHeatmapPoints(
  includeZeroGps: boolean = false
): Promise<HeatmapPoint[]> {
  const data = await apiFetch<HeatmapResponse>(
    `/collection/heatmap?include_zero_gps=${includeZeroGps}`
  )
  return data.points ?? []
}

export async function fetchCollectionViolationDetail(
  violationId: string
): Promise<CollectionViolationDetailResponse> {
  return apiFetch<CollectionViolationDetailResponse>(
    `/collection/violations/${encodeURIComponent(violationId)}`
  )
}

export async function verifyViolation(
  request: VerifyViolationRequest
): Promise<VerifyViolationResponse> {
  return apiFetch<VerifyViolationResponse>('/collection/verify', {
    method: 'POST',
    body: JSON.stringify(request),
    headers: { 'Content-Type': 'application/json' },
  })
}

// SWR key factories (prevents magic strings across components)
export const SWR_KEYS = {
  violations: '/violations',
  stats: '/stats',
  health: '/health',
} as const
