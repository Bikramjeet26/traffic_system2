'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Upload, ImageIcon, Loader2, CheckCircle2, AlertCircle,
  MapPin, Clock, Car, CreditCard, ShieldAlert, X, Crosshair, Users,
  Video, Minus, RotateCcw,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Progress } from '@/components/ui/progress'
import {
  analyzeImage,
  analyzeVideo,
  evidenceAssetUrl,
  evidenceImageUrl,
  violationLabel,
  type AnalyzeResponse,
  type AnalyzeVideoResponse,
} from '@/lib/api'
import { cn } from '@/lib/utils'

type AnalyzeMode = 'image' | 'video'
type NormalizedPoint = { x: number; y: number }

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

function ModeToggle({
  mode,
  onChange,
}: {
  mode: AnalyzeMode
  onChange: (mode: AnalyzeMode) => void
}) {
  return (
    <div className="flex rounded-lg border border-border bg-muted/30 p-0.5 w-full sm:w-auto">
      <Button
        type="button"
        variant={mode === 'image' ? 'default' : 'ghost'}
        size="sm"
        className="flex-1 sm:flex-none h-8 text-xs gap-1.5"
        onClick={() => onChange('image')}
      >
        <ImageIcon className="size-3.5" />
        Image
      </Button>
      <Button
        type="button"
        variant={mode === 'video' ? 'default' : 'ghost'}
        size="sm"
        className="flex-1 sm:flex-none h-8 text-xs gap-1.5"
        onClick={() => onChange('video')}
      >
        <Video className="size-3.5" />
        Video Stop / Red Light
      </Button>
    </div>
  )
}

export function AnalyzePage() {
  const [mode, setMode] = useState<AnalyzeMode>('image')

  // ── Image mode state ──
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // ── Video mode state ──
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoPreview, setVideoPreview] = useState<string | null>(null)
  const [videoResult, setVideoResult] = useState<AnalyzeVideoResponse | null>(null)
  const [stopLinePoints, setStopLinePoints] = useState<NormalizedPoint[]>([])
  const [lightState, setLightState] = useState<'red' | 'green'>('red')
  const [videoReady, setVideoReady] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)

  // ── Shared state ──
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const [gps, setGps] = useState<{ lat: number; lon: number } | null>(null)
  const [capturedAt, setCapturedAt] = useState<string | null>(null)
  const [geoStatus, setGeoStatus] = useState<'idle' | 'locating' | 'success' | 'error'>('idle')
  const [geoError, setGeoError] = useState<string | null>(null)

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
      if (videoPreview) URL.revokeObjectURL(videoPreview)
    }
  }, [preview, videoPreview])

  const resetShared = () => {
    setError(null)
    setProgress(0)
  }

  const switchMode = (next: AnalyzeMode) => {
    setMode(next)
    resetShared()
    setLoading(false)
  }

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

  const pickImageFile = (picked: File) => {
    setFile(picked)
    setResult(null)
    resetShared()
    if (preview) URL.revokeObjectURL(preview)
    setPreview(URL.createObjectURL(picked))
  }

  const pickVideoFile = (picked: File) => {
    setVideoFile(picked)
    setVideoResult(null)
    setStopLinePoints([])
    setVideoReady(false)
    resetShared()
    if (videoPreview) URL.revokeObjectURL(videoPreview)
    setVideoPreview(URL.createObjectURL(picked))
  }

  const handleImageFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) pickImageFile(e.target.files[0])
  }

  const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) pickVideoFile(e.target.files[0])
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (!dropped) return
    if (mode === 'image' && dropped.type.startsWith('image/')) pickImageFile(dropped)
    if (mode === 'video' && dropped.type.startsWith('video/')) pickVideoFile(dropped)
  }, [mode])

  const clearImageFile = () => {
    setFile(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setResult(null)
    resetShared()
    if (inputRef.current) inputRef.current.value = ''
  }

  const clearVideoFile = () => {
    setVideoFile(null)
    if (videoPreview) URL.revokeObjectURL(videoPreview)
    setVideoPreview(null)
    setVideoResult(null)
    setStopLinePoints([])
    setVideoReady(false)
    resetShared()
    if (videoInputRef.current) videoInputRef.current.value = ''
  }

  const handleVideoLoaded = () => {
    const video = videoRef.current
    if (!video) return
    video.pause()
    video.currentTime = 0
    setVideoReady(true)
  }

  const handleLineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current
    if (!video || !video.videoWidth || loading) return

    const rect = video.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * video.videoWidth
    const y = ((e.clientY - rect.top) / rect.height) * video.videoHeight

    const point: NormalizedPoint = {
      x: x / video.videoWidth,
      y: y / video.videoHeight,
    }

    setStopLinePoints(prev => {
      if (prev.length >= 4) return [point]
      return [...prev, point]
    })
  }

  const clearStopLine = () => setStopLinePoints([])

  const linesComplete = stopLinePoints.length === 4
  const stopLineNorm: [number, number, number, number] | null = linesComplete
    ? [stopLinePoints[0].x, stopLinePoints[0].y, stopLinePoints[1].x, stopLinePoints[1].y]
    : null
  const redLightLineNorm: [number, number, number, number] | null = linesComplete
    ? [stopLinePoints[2].x, stopLinePoints[2].y, stopLinePoints[3].x, stopLinePoints[3].y]
    : null

  const runImageAnalysis = async () => {
    if (!file) return
    setLoading(true)
    resetShared()
    setProgress(10)

    const ticker = setInterval(() => {
      setProgress(p => Math.min(p + 8, 85))
    }, 600)

    try {
      const ts = capturedAt ?? new Date().toISOString()
      const data = await analyzeImage(file, {
        timestamp: ts,
        gpsLat: gps?.lat,
        gpsLon: gps?.lon,
      })
      setProgress(100)
      setResult(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      clearInterval(ticker)
      setLoading(false)
    }
  }

  const runVideoAnalysis = async () => {
    if (!videoFile || !stopLineNorm || !redLightLineNorm) return
    setLoading(true)
    resetShared()
    setProgress(5)

    const ticker = setInterval(() => {
      setProgress(p => Math.min(p + 3, 90))
    }, 1500)

    try {
      const data = await analyzeVideo(videoFile, {
        stopLine: stopLineNorm,
        redLightLine: redLightLineNorm,
        initialLightState: lightState,
      })
      setProgress(100)
      setVideoResult(data)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error')
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

  const annotatedVideoSrc = videoResult?.annotated_video_url
    ? evidenceAssetUrl(videoResult.annotated_video_url)
    : null

  return (
    <div className="flex flex-col gap-4">
      <ModeToggle mode={mode} onChange={switchMode} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Left: Upload Panel ── */}
        <div className="flex flex-col gap-4">
          {mode === 'image' ? (
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
                    onChange={handleImageFileChange}
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
                        type="button"
                        onClick={e => { e.stopPropagation(); clearImageFile() }}
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

                {file && (
                  <div className="mt-3 flex items-center justify-between rounded-md bg-muted/30 px-3 py-2 border border-border">
                    <span className="text-xs text-muted-foreground truncate max-w-[220px]">{file.name}</span>
                    <span className="text-xs text-muted-foreground ml-2 shrink-0">
                      {(file.size / 1024).toFixed(0)} KB
                    </span>
                  </div>
                )}

                {loading && mode === 'image' && (
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
                  onClick={runImageAnalysis}
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
          ) : (
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Video className="size-4 text-primary" />
                  Upload Video &amp; Draw Lines
                </CardTitle>
                <CardDescription className="text-muted-foreground text-sm">
                  Upload a traffic video, click four points on the first frame (stop line then red light line), then run detection.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <div
                  onDragOver={e => { e.preventDefault(); setDragging(true) }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => !videoPreview && videoInputRef.current?.click()}
                  className={cn(
                    'relative rounded-lg border-2 border-dashed transition-colors overflow-hidden',
                    dragging
                      ? 'border-primary bg-primary/5'
                      : videoPreview
                        ? 'border-border bg-muted/20'
                        : 'border-border hover:border-primary/50 hover:bg-muted/10 cursor-pointer min-h-[220px] flex items-center justify-center'
                  )}
                >
                  <input
                    ref={videoInputRef}
                    type="file"
                    accept="video/mp4,video/quicktime,video/webm,video/*"
                    className="sr-only"
                    onChange={handleVideoFileChange}
                  />

                  {videoPreview ? (
                    <>
                      <div
                        className="relative w-full cursor-crosshair"
                        onClick={handleLineClick}
                        role="presentation"
                      >
                        <video
                          ref={videoRef}
                          src={videoPreview}
                          className="max-h-[320px] w-full object-contain bg-black/80"
                          muted
                          playsInline
                          onLoadedMetadata={handleVideoLoaded}
                        />
                        {videoReady && (
                          <svg
                            className="absolute inset-0 w-full h-full pointer-events-none"
                            viewBox="0 0 100 100"
                            preserveAspectRatio="none"
                          >
                            {stopLinePoints.map((p, i) => (
                              <circle
                                key={i}
                                cx={p.x * 100}
                                cy={p.y * 100}
                                r={0.8}
                                fill={i < 2 ? '#22c55e' : '#ef4444'}
                                stroke="white"
                                strokeWidth={0.15}
                              />
                            ))}
                            {stopLinePoints.length >= 2 && (
                              <line
                                x1={stopLinePoints[0].x * 100}
                                y1={stopLinePoints[0].y * 100}
                                x2={stopLinePoints[1].x * 100}
                                y2={stopLinePoints[1].y * 100}
                                stroke="#22c55e"
                                strokeWidth={0.35}
                              />
                            )}
                            {stopLinePoints.length === 4 && (
                              <line
                                x1={stopLinePoints[2].x * 100}
                                y1={stopLinePoints[2].y * 100}
                                x2={stopLinePoints[3].x * 100}
                                y2={stopLinePoints[3].y * 100}
                                stroke="#ef4444"
                                strokeWidth={0.35}
                              />
                            )}
                          </svg>
                        )}
                        {!linesComplete && videoReady && (
                          <div className="absolute bottom-2 left-2 right-2 rounded bg-black/60 px-2 py-1 text-[11px] text-white text-center">
                            {stopLinePoints.length < 2
                              ? `Click 2 points for stop line (${stopLinePoints.length}/2)`
                              : stopLinePoints.length < 4
                                ? `Click 2 points for red light line (${stopLinePoints.length - 2}/2)`
                                : ''}
                          </div>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={e => { e.stopPropagation(); clearVideoFile() }}
                        className="absolute top-2 right-2 flex size-6 items-center justify-center rounded-full bg-card/80 border border-border hover:bg-destructive/20 transition-colors z-10"
                        aria-label="Remove video"
                      >
                        <X className="size-3 text-foreground" />
                      </button>
                    </>
                  ) : (
                    <div className="flex flex-col items-center gap-3 px-6 text-center py-8">
                      <div className="flex size-12 items-center justify-center rounded-full bg-muted border border-border">
                        <Video className="size-6 text-muted-foreground" />
                      </div>
                      <div>
                        <p className="text-sm text-foreground font-medium">Drop video here or click to browse</p>
                        <p className="text-xs text-muted-foreground mt-1">Supports MP4, MOV, WebM</p>
                      </div>
                    </div>
                  )}
                </div>

                {videoFile && (
                  <div className="flex items-center justify-between rounded-md bg-muted/30 px-3 py-2 border border-border">
                    <span className="text-xs text-muted-foreground truncate max-w-[220px]">{videoFile.name}</span>
                    <span className="text-xs text-muted-foreground ml-2 shrink-0">
                      {(videoFile.size / (1024 * 1024)).toFixed(1)} MB
                    </span>
                  </div>
                )}

                <div className="flex flex-col gap-2">
                  <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Traffic Light</p>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant={lightState === 'red' ? 'default' : 'outline'}
                      size="sm"
                      className="flex-1 text-xs"
                      onClick={() => setLightState('red')}
                    >
                      Red (violations on)
                    </Button>
                    <Button
                      type="button"
                      variant={lightState === 'green' ? 'default' : 'outline'}
                      size="sm"
                      className="flex-1 text-xs"
                      onClick={() => setLightState('green')}
                    >
                      Green (no violations)
                    </Button>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="flex-1 text-xs gap-1.5"
                    onClick={clearStopLine}
                    disabled={stopLinePoints.length === 0 || loading}
                  >
                    <RotateCcw className="size-3.5" />
                    Clear Lines
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="flex-1 text-xs gap-1.5"
                    onClick={() => videoInputRef.current?.click()}
                    disabled={loading}
                  >
                    Change Video
                  </Button>
                </div>

                {loading && mode === 'video' && (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Processing video on backend…</span>
                      <span className="text-xs text-primary">{progress}%</span>
                    </div>
                    <Progress value={progress} className="h-1.5" />
                    <p className="text-[11px] text-muted-foreground">Long videos may take several minutes.</p>
                  </div>
                )}

                <Button
                  className="w-full"
                  onClick={runVideoAnalysis}
                  disabled={!videoFile || !linesComplete || !videoReady || loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="size-4 animate-spin" data-icon="inline-start" />
                      Processing Video…
                    </>
                  ) : (
                    <>
                      <Minus className="size-4" data-icon="inline-start" />
                      Run Detection
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          )}

          {mode === 'image' && (
            <>
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
            </>
          )}

          {mode === 'video' && (
            <Card className="bg-card border-border">
              <CardContent className="pt-4">
                <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-3">Stop Line &amp; Red Light Pipeline</p>
                <div className="flex flex-col gap-2 text-xs text-muted-foreground">
                  <p>1. Upload a traffic video (MP4 recommended).</p>
                  <p>2. Click two points for the stop line, then two for the red light line.</p>
                  <p>3. Set traffic light state for the clip.</p>
                  <p>4. Run detection — stop line and red light crossings are flagged.</p>
                </div>
              </CardContent>
            </Card>
          )}
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

          {mode === 'image' && !result && !loading && !error && (
            <Card className="bg-card border-border border-dashed min-h-[320px] flex items-center justify-center">
              <div className="flex flex-col items-center gap-3 text-center px-8">
                <div className="flex size-14 items-center justify-center rounded-full bg-muted border border-border">
                  <ShieldAlert className="size-7 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground">Upload an image and run analysis to see detection results here.</p>
              </div>
            </Card>
          )}

          {mode === 'video' && !videoResult && !loading && !error && (
            <Card className="bg-card border-border border-dashed min-h-[320px] flex items-center justify-center">
              <div className="flex flex-col items-center gap-3 text-center px-8">
                <div className="flex size-14 items-center justify-center rounded-full bg-muted border border-border">
                  <Video className="size-7 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground">
                  Upload a video, draw both lines, and run detection to see annotated output here.
                </p>
              </div>
            </Card>
          )}

          {loading && ((mode === 'image' && !result) || (mode === 'video' && !videoResult)) && (
            <Card className="bg-card border-border min-h-[320px] flex items-center justify-center">
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="size-10 text-primary animate-spin" />
                <p className="text-sm text-muted-foreground">
                  {mode === 'video' ? 'Processing video frames…' : 'Running detection pipeline…'}
                </p>
              </div>
            </Card>
          )}

          {mode === 'image' && result && (
            <>
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

          {mode === 'video' && videoResult && (
            <>
              {(() => {
                const stopLineCount = videoResult.violations.filter(v => v.type === 'stop_line_violation').length
                const redLightCount = videoResult.violations.filter(v => v.type === 'red_light_violation').length
                return (
              <div className="grid grid-cols-3 gap-3">
                <Card className="bg-card border-border">
                  <CardContent className="pt-4 pb-3 text-center">
                    <p className="text-2xl font-bold text-foreground">{stopLineCount}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">Stop Line</p>
                  </CardContent>
                </Card>
                <Card className="bg-card border-border">
                  <CardContent className="pt-4 pb-3 text-center">
                    <p className="text-2xl font-bold text-foreground">{redLightCount}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">Red Light</p>
                  </CardContent>
                </Card>
                <Card className={cn(
                  'border',
                  videoResult.violations_count > 0
                    ? 'bg-[--violation]/5 border-[--violation]/20'
                    : 'bg-[--safe]/5 border-[--safe]/20'
                )}>
                  <CardContent className="pt-4 pb-3 text-center">
                    <div className="flex justify-center mb-1">
                      {videoResult.violations_count > 0
                        ? <AlertCircle className="size-5 text-[--violation]" />
                        : <CheckCircle2 className="size-5 text-[--safe]" />}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {videoResult.violations_count > 0 ? `${videoResult.violations_count} Total` : 'No Violations'}
                    </p>
                  </CardContent>
                </Card>
              </div>
                )
              })()}

              {annotatedVideoSrc && (
                <Card className="bg-card border-border overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Video className="size-4 text-primary" />
                      Annotated Output Video
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <video
                      key={annotatedVideoSrc}
                      src={annotatedVideoSrc}
                      controls
                      playsInline
                      preload="metadata"
                      className="w-full max-h-[360px] bg-black object-contain"
                    />
                    <div className="px-4 py-2 border-t border-border">
                      <a
                        href={annotatedVideoSrc}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary hover:underline"
                      >
                        Open video in new tab
                      </a>
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card className="bg-card border-border">
                <CardContent className="pt-4 flex flex-col gap-3">
                  {videoResult.violations.length > 0 ? (
                    <>
                      <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                        Violations ({videoResult.violations.length})
                      </p>
                      <div className="flex flex-col gap-2">
                        {videoResult.violations.map((v, i) => (
                          <div
                            key={`${v.track_id}-${v.frame_idx}-${i}`}
                            className="flex items-center justify-between rounded-md bg-[--violation]/5 px-3 py-2 border border-[--violation]/15 text-xs"
                          >
                            <div className="flex items-center gap-2">
                              <ViolationBadge type={v.type ?? 'stop_line_violation'} />
                              <span className="text-muted-foreground">Track #{v.track_id}</span>
                            </div>
                            <div className="flex items-center gap-2 text-muted-foreground font-mono">
                              <span>Frame {v.frame_idx}</span>
                              <Badge variant="outline" className="text-[10px] h-4 px-1.5 uppercase">
                                {v.light_state}
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No stop line or red light violations detected.
                    </p>
                  )}

                  {videoResult.results_json_url && (
                    <>
                      <Separator />
                      <a
                        href={evidenceAssetUrl(videoResult.results_json_url)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-primary hover:underline"
                      >
                        Download full results JSON
                      </a>
                    </>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
