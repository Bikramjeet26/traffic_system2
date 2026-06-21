'use client'

import { useCallback, useRef, useState } from 'react'
import {
  Upload, ImageIcon, Loader2, CheckCircle2, AlertCircle,
  MapPin, Clock, Car, CreditCard, ShieldAlert, X, Crosshair, Users
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import { analyzeImage, evidenceImageUrl, violationLabel, type AnalyzeResponse } from '@/lib/api'
import { cn } from '@/lib/utils'

function ViolationBadge({ type }: { type: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium bg-[--violation]/15 text-[--violation] border border-[--violation]/20">
      <ShieldAlert className="size-3" />
      {violationLabel(type)}
    </span>
  )
}

function PlateBadge({ text }: { text: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded px-2.5 py-0.5 text-xs font-mono font-semibold bg-[--plate]/15 text-[--plate] border border-[--plate]/20">
      <CreditCard className="size-3" />
      {text}
    </span>
  )
}

export function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // GPS + time capture (stored client-side, sent with the analysis request)
  const [gps, setGps] = useState<{ lat: number; lon: number } | null>(null)
  const [capturedAt, setCapturedAt] = useState<string | null>(null)
  const [geoStatus, setGeoStatus] = useState<'idle' | 'locating' | 'success' | 'error'>('idle')
  const [geoError, setGeoError] = useState<string | null>(null)

  const captureLocation = () => {
    setGeoError(null)
    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      setGeoStatus('error')
      setGeoError('Geolocation is not supported by this browser.')
      return
    }
    setGeoStatus('locating')
    navigator.geolocation.getCurrentPosition(
      pos => {
        setGps({ lat: pos.coords.latitude, lon: pos.coords.longitude })
        setCapturedAt(new Date().toISOString())
        setGeoStatus('success')
      },
      err => {
        setGeoStatus('error')
        setGeoError(
          err.code === err.PERMISSION_DENIED
            ? 'Location permission denied. Allow access to capture GPS.'
            : 'Unable to retrieve your location.'
        )
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    )
  }

  const clearLocation = () => {
    setGps(null)
    setCapturedAt(null)
    setGeoStatus('idle')
    setGeoError(null)
  }

  const pickFile = (picked: File) => {
    setFile(picked)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(picked)
    setPreview(url)
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) pickFile(e.target.files[0])
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped && dropped.type.startsWith('image/')) pickFile(dropped)
  }, [])

  const clearFile = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const runAnalysis = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setProgress(10)

    // Simulate progress ticks while waiting
    const ticker = setInterval(() => {
      setProgress(p => Math.min(p + 8, 85))
    }, 600)

    try {
      // Use the captured timestamp if the user grabbed GPS, else current time
      const ts = capturedAt ?? new Date().toISOString()
      const data = await analyzeImage(file, {
        timestamp: ts,
        gpsLat: gps?.lat,
        gpsLon: gps?.lon,
      })
      setProgress(100)
      setResult(data)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setError(msg)
    } finally {
      clearInterval(ticker)
      setLoading(false)
    }
  }

  const annotatedSrc = result?.annotated_image_base64
    ? `data:image/jpeg;base64,${result.annotated_image_base64}`
    : result?.evidence?.annotated_image_path
      ? evidenceImageUrl(result.evidence.annotated_image_path)
      : null

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* ── Left: Upload Panel ── */}
      <div className="flex flex-col gap-4">
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Upload className="size-4 text-primary" />
              Upload Traffic Image
            </CardTitle>
            <CardDescription className="text-muted-foreground text-sm">
              Upload a road scene image to run the full violation detection pipeline.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              className={cn(
                'relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed cursor-pointer transition-colors min-h-[220px]',
                dragging
                  ? 'border-primary bg-primary/5'
                  : preview
                    ? 'border-border bg-muted/20'
                    : 'border-border hover:border-primary/50 hover:bg-muted/10'
              )}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={handleFileChange}
              />
              {preview ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={preview}
                    alt="Selected image preview"
                    className="max-h-[280px] w-full object-contain rounded-lg"
                  />
                  <button
                    onClick={e => { e.stopPropagation(); clearFile() }}
                    className="absolute top-2 right-2 flex size-6 items-center justify-center rounded-full bg-card/80 border border-border hover:bg-destructive/20 transition-colors"
                    aria-label="Remove image"
                  >
                    <X className="size-3 text-foreground" />
                  </button>
                </>
              ) : (
                <div className="flex flex-col items-center gap-3 px-6 text-center">
                  <div className="flex size-12 items-center justify-center rounded-full bg-muted border border-border">
                    <ImageIcon className="size-6 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="text-sm text-foreground font-medium">Drop image here or click to browse</p>
                    <p className="text-xs text-muted-foreground mt-1">Supports JPEG, PNG, WebP</p>
                  </div>
                </div>
              )}
            </div>

            {/* File info */}
            {file && (
              <div className="mt-3 flex items-center justify-between rounded-md bg-muted/30 px-3 py-2 border border-border">
                <span className="text-xs text-muted-foreground truncate max-w-[220px]">{file.name}</span>
                <span className="text-xs text-muted-foreground ml-2 shrink-0">
                  {(file.size / 1024).toFixed(0)} KB
                </span>
              </div>
            )}

            {/* Progress bar */}
            {loading && (
              <div className="mt-3 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Running pipeline…</span>
                  <span className="text-xs text-primary">{progress}%</span>
                </div>
                <Progress value={progress} className="h-1.5" />
              </div>
            )}

            <Button
              className="mt-4 w-full"
              onClick={runAnalysis}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <Loader2 className="size-4 animate-spin" data-icon="inline-start" />
                  Analyzing…
                </>
              ) : (
                <>
                  <ShieldAlert className="size-4" data-icon="inline-start" />
                  Analyze Image
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Location & time capture */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <MapPin className="size-4 text-primary" />
              Location &amp; Time
            </CardTitle>
            <CardDescription className="text-muted-foreground text-sm">
              Tag this evidence with your current GPS coordinates and capture time.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="flex-1"
                onClick={captureLocation}
                disabled={geoStatus === 'locating'}
              >
                {geoStatus === 'locating' ? (
                  <>
                    <Loader2 className="size-4 animate-spin" data-icon="inline-start" />
                    Locating…
                  </>
                ) : (
                  <>
                    <Crosshair className="size-4" data-icon="inline-start" />
                    {gps ? 'Update GPS & Time' : 'Capture GPS & Time'}
                  </>
                )}
              </Button>
              {gps && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={clearLocation}
                  aria-label="Clear captured location"
                >
                  <X className="size-4" />
                </Button>
              )}
            </div>

            {geoError && (
              <p className="text-xs text-destructive flex items-center gap-1.5">
                <AlertCircle className="size-3.5 shrink-0" />
                {geoError}
              </p>
            )}

            {gps && (
              <div className="flex flex-col gap-2 rounded-md bg-muted/30 border border-border px-3 py-2.5">
                <div className="flex items-center gap-2 text-xs text-foreground">
                  <MapPin className="size-3.5 text-primary shrink-0" />
                  <span className="font-mono">{gps.lat.toFixed(6)}, {gps.lon.toFixed(6)}</span>
                </div>
                {capturedAt && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Clock className="size-3.5 shrink-0" />
                    <span className="font-mono">{new Date(capturedAt).toLocaleString()}</span>
                  </div>
                )}
              </div>
            )}

            {!gps && !geoError && (
              <p className="text-xs text-muted-foreground">
                Optional — if skipped, the current time is used and GPS defaults to 0, 0.
              </p>
            )}
          </CardContent>
        </Card>

        {/* Pipeline steps info card */}
        <Card className="bg-card border-border">
          <CardContent className="pt-4">
            <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-3">Detection Pipeline</p>
            <div className="flex flex-col gap-2">
              {[
                { icon: Car, label: 'Vehicle Detection', desc: 'BDD100K YOLOv11' },
                { icon: CreditCard, label: 'License Plate OCR', desc: 'RapidOCR + YOLOv11' },
                { icon: ShieldAlert, label: 'Helmet Compliance', desc: 'helmet_detection.pt' },
                { icon: ShieldAlert, label: 'Seatbelt Compliance', desc: 'seatbelt_detection.pt' },
                { icon: Users, label: 'Triple Riding', desc: '3+ riders on two-wheeler' },
              ].map(({ icon: Icon, label, desc }) => (
                <div key={label} className="flex items-center gap-3">
                  <div className="flex size-7 shrink-0 items-center justify-center rounded bg-muted border border-border">
                    <Icon className="size-3.5 text-muted-foreground" />
                  </div>
                  <div className="flex flex-col leading-tight">
                    <span className="text-xs font-medium text-foreground">{label}</span>
                    <span className="text-[11px] text-muted-foreground">{desc}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── Right: Results Panel ── */}
      <div className="flex flex-col gap-4">
        {error && (
          <Card className="border-destructive/30 bg-destructive/5">
            <CardContent className="pt-4 flex items-start gap-3">
              <AlertCircle className="size-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-destructive">Analysis Failed</p>
                <p className="text-xs text-muted-foreground mt-1">{error}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {!result && !loading && !error && (
          <Card className="bg-card border-border border-dashed min-h-[320px] flex items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-center px-8">
              <div className="flex size-14 items-center justify-center rounded-full bg-muted border border-border">
                <ShieldAlert className="size-7 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground">Upload an image and run analysis to see detection results here.</p>
            </div>
          </Card>
        )}

        {loading && !result && (
          <Card className="bg-card border-border min-h-[320px] flex items-center justify-center">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="size-10 text-primary animate-spin" />
              <p className="text-sm text-muted-foreground">Running detection pipeline…</p>
            </div>
          </Card>
        )}

        {result && (
          <>
            {/* Summary stats */}
            <div className="grid grid-cols-3 gap-3">
              <Card className="bg-card border-border">
                <CardContent className="pt-4 pb-3 text-center">
                  <p className="text-2xl font-bold text-foreground">{result.violations_count}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">Violations</p>
                </CardContent>
              </Card>
              <Card className="bg-card border-border">
                <CardContent className="pt-4 pb-3 text-center">
                  <p className="text-2xl font-bold text-foreground">{result.plates_count}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">Plates Read</p>
                </CardContent>
              </Card>
              <Card className={cn('border', result.violations_count > 0 ? 'bg-[--violation]/5 border-[--violation]/20' : 'bg-[--safe]/5 border-[--safe]/20')}>
                <CardContent className="pt-4 pb-3 text-center">
                  <div className="flex justify-center mb-1">
                    {result.violations_count > 0
                      ? <AlertCircle className="size-5 text-[--violation]" />
                      : <CheckCircle2 className="size-5 text-[--safe]" />}
                  </div>
                  <p className="text-xs text-muted-foreground">{result.violations_count > 0 ? 'Flagged' : 'Clean'}</p>
                </CardContent>
              </Card>
            </div>

            {/* Annotated image */}
            {annotatedSrc && (
              <Card className="bg-card border-border overflow-hidden">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <CheckCircle2 className="size-4 text-[--safe]" />
                    Annotated Evidence
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={annotatedSrc}
                    alt="Annotated traffic scene with bounding boxes"
                    className="w-full object-contain max-h-[320px]"
                  />
                </CardContent>
              </Card>
            )}

            {/* Meta */}
            <Card className="bg-card border-border">
              <CardContent className="pt-4 flex flex-col gap-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock className="size-3.5" />
                  <span className="font-mono">{new Date(result.timestamp).toLocaleString()}</span>
                </div>
                {(result.gps.latitude !== 0 || result.gps.longitude !== 0) && (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <MapPin className="size-3.5" />
                    <span className="font-mono">{result.gps.latitude.toFixed(4)}, {result.gps.longitude.toFixed(4)}</span>
                  </div>
                )}

                {result.evidence.plates.length > 0 && (
                  <>
                    <Separator />
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-2">License Plates</p>
                      <div className="flex flex-wrap gap-2">
                        {result.evidence.plates.map((p, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <PlateBadge text={p.text || '—'} />
                            <span className="text-[11px] text-muted-foreground">{(p.confidence * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}

                {result.evidence.violations.length > 0 && (
                  <>
                    <Separator />
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-2">Violations Detected</p>
                      <div className="flex flex-col gap-2">
                        {result.evidence.violations.map((v, i) => (
                          <div key={i} className="flex items-center justify-between rounded-md bg-muted/20 px-3 py-1.5 border border-border">
                            <ViolationBadge type={v.type} />
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              {v.person_count != null && (
                                <span className="flex items-center gap-1">
                                  <Users className="size-3" />
                                  {v.person_count}
                                </span>
                              )}
                              <Car className="size-3" />
                              <span>{v.vehicle_class}</span>
                              <Badge variant="outline" className="text-[10px] h-4 px-1.5 font-mono">
                                {(v.confidence * 100).toFixed(0)}%
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
