'use client'

import { useRef, useState } from 'react'
import {
  ScanFace, Upload, Loader2, AlertCircle, CheckCircle2, XCircle,
  X, UserSearch, Database, ImageIcon, MapPin, Clock
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  faceMatchCompare,
  faceMatchViolation,
  evidenceImageUrl,
  type FaceCompareResult,
  type FaceScanResult,
} from '@/lib/api'
import { cn } from '@/lib/utils'

// ── Reusable image picker ─────────────────────────────────────────────────────

function ImagePicker({
  label,
  file,
  preview,
  onPick,
  onClear,
}: {
  label: string
  file: File | null
  preview: string | null
  onPick: (f: File) => void
  onClear: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">{label}</p>
      <div
        onClick={() => inputRef.current?.click()}
        className={cn(
          'relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed cursor-pointer transition-colors min-h-[160px]',
          preview ? 'border-border bg-muted/20' : 'border-border hover:border-primary/50 hover:bg-muted/10'
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={e => { if (e.target.files?.[0]) onPick(e.target.files[0]) }}
        />
        {preview ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt={`${label} preview`} className="max-h-[200px] w-full object-contain rounded-lg" />
            <button
              onClick={e => { e.stopPropagation(); onClear() }}
              className="absolute top-2 right-2 flex size-6 items-center justify-center rounded-full bg-card/80 border border-border hover:bg-destructive/20 transition-colors"
              aria-label={`Remove ${label}`}
            >
              <X className="size-3 text-foreground" />
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 px-4 text-center">
            <div className="flex size-10 items-center justify-center rounded-full bg-muted border border-border">
              <ImageIcon className="size-5 text-muted-foreground" />
            </div>
            <p className="text-xs text-muted-foreground">Click to upload</p>
          </div>
        )}
      </div>
      {file && <p className="text-[11px] text-muted-foreground truncate">{file.name}</p>}
    </div>
  )
}

function MatchVerdict({ isMatch, similarity }: { isMatch: boolean; similarity?: number }) {
  return (
    <div
      className={cn(
        'flex items-center gap-3 rounded-lg border px-4 py-3',
        isMatch ? 'bg-[--violation]/5 border-[--violation]/30' : 'bg-[--safe]/5 border-[--safe]/30'
      )}
    >
      {isMatch ? (
        <AlertCircle className="size-6 text-[--violation] shrink-0" />
      ) : (
        <CheckCircle2 className="size-6 text-[--safe] shrink-0" />
      )}
      <div className="flex-1">
        <p className={cn('text-sm font-semibold', isMatch ? 'text-[--violation]' : 'text-[--safe]')}>
          {isMatch ? 'Match Found' : 'No Match'}
        </p>
        {similarity != null && (
          <p className="text-xs text-muted-foreground">
            Similarity {(similarity * 100).toFixed(1)}%
          </p>
        )}
      </div>
    </div>
  )
}

// ── Mode 1: Compare two images ─────────────────────────────────────────────────

function CompareMode() {
  const [person, setPerson] = useState<{ file: File | null; preview: string | null }>({ file: null, preview: null })
  const [target, setTarget] = useState<{ file: File | null; preview: string | null }>({ file: null, preview: null })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FaceCompareResult | null>(null)

  const pick = (
    set: typeof setPerson,
  ) => (f: File) => {
    set({ file: f, preview: URL.createObjectURL(f) })
    setResult(null)
    setError(null)
  }
  const clear = (set: typeof setPerson) => () => set({ file: null, preview: null })

  const run = async () => {
    if (!person.file || !target.file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await faceMatchCompare(person.file, target.file)
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <ImagePicker
          label="Reference / Criminal"
          file={person.file}
          preview={person.preview}
          onPick={pick(setPerson)}
          onClear={clear(setPerson)}
        />
        <ImagePicker
          label="Image to Compare"
          file={target.file}
          preview={target.preview}
          onPick={pick(setTarget)}
          onClear={clear(setTarget)}
        />
      </div>

      <Button onClick={run} disabled={!person.file || !target.file || loading}>
        {loading ? (
          <><Loader2 className="size-4 animate-spin" data-icon="inline-start" />Comparing…</>
        ) : (
          <><ScanFace className="size-4" data-icon="inline-start" />Compare Faces</>
        )}
      </Button>

      {error && (
        <div className="flex items-start gap-2 rounded-md bg-destructive/5 border border-destructive/30 px-3 py-2">
          <AlertCircle className="size-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-xs text-muted-foreground">{error}</p>
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-3">
          <MatchVerdict isMatch={result.is_match} similarity={result.best_match?.similarity} />
          <div className="grid grid-cols-2 gap-3 text-xs text-muted-foreground">
            <div className="rounded-md bg-muted/30 border border-border px-3 py-2">
              Faces in reference: <span className="text-foreground font-medium">{result.faces_in_person ?? '—'}</span>
            </div>
            <div className="rounded-md bg-muted/30 border border-border px-3 py-2">
              Faces in target: <span className="text-foreground font-medium">{result.faces_in_target ?? '—'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Mode 2: Scan stored violations ─────────────────────────────────────────────

function ScanMode() {
  const [person, setPerson] = useState<{ file: File | null; preview: string | null }>({ file: null, preview: null })
  const [violationId, setViolationId] = useState('')
  const [scanAll, setScanAll] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FaceScanResult | null>(null)

  const pick = (f: File) => {
    setPerson({ file: f, preview: URL.createObjectURL(f) })
    setResult(null)
    setError(null)
  }

  const run = async () => {
    if (!person.file) return
    if (!scanAll && !violationId.trim()) {
      setError('Enter a violation id or switch to "Scan entire database".')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await faceMatchViolation(person.file, {
        scanAll,
        violationId: scanAll ? undefined : violationId.trim(),
      })
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan failed')
    } finally {
      setLoading(false)
    }
  }

  const matches = result?.all_results ?? []

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <ImagePicker
          label="Reference / Criminal"
          file={person.file}
          preview={person.preview}
          onPick={pick}
          onClear={() => setPerson({ file: null, preview: null })}
        />
        <div className="flex flex-col gap-3">
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">Search Scope</p>
          <div className="flex flex-col gap-2 rounded-lg border border-border bg-muted/10 p-3">
            <button
              onClick={() => setScanAll(true)}
              className={cn(
                'flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors text-left',
                scanAll ? 'bg-primary/10 text-primary border border-primary/20' : 'text-muted-foreground hover:bg-muted/40 border border-transparent'
              )}
            >
              <Database className="size-4" />
              Scan entire violation database
            </button>
            <button
              onClick={() => setScanAll(false)}
              className={cn(
                'flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors text-left',
                !scanAll ? 'bg-primary/10 text-primary border border-primary/20' : 'text-muted-foreground hover:bg-muted/40 border border-transparent'
              )}
            >
              <UserSearch className="size-4" />
              Match a specific violation
            </button>
            {!scanAll && (
              <input
                value={violationId}
                onChange={e => setViolationId(e.target.value)}
                placeholder="Violation id / timestamp"
                className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-xs font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            )}
          </div>
        </div>
      </div>

      <Button onClick={run} disabled={!person.file || loading}>
        {loading ? (
          <><Loader2 className="size-4 animate-spin" data-icon="inline-start" />Scanning…</>
        ) : (
          <><ScanFace className="size-4" data-icon="inline-start" />{scanAll ? 'Scan Violations' : 'Match Violation'}</>
        )}
      </Button>

      {error && (
        <div className="flex items-start gap-2 rounded-md bg-destructive/5 border border-destructive/30 px-3 py-2">
          <AlertCircle className="size-4 text-destructive shrink-0 mt-0.5" />
          <p className="text-xs text-muted-foreground">{error}</p>
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-3">
          <MatchVerdict
            isMatch={result.is_match}
            similarity={
              typeof result.best_match?.similarity === 'number' ? result.best_match.similarity : undefined
            }
          />

          {result.mode === 'violation_database_scan' && (
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="rounded-md bg-muted/30 border border-border px-2 py-2">
                <p className="text-lg font-bold text-foreground">{result.total_violations_scanned ?? 0}</p>
                <p className="text-[11px] text-muted-foreground">Scanned</p>
              </div>
              <div className="rounded-md bg-muted/30 border border-border px-2 py-2">
                <p className="text-lg font-bold text-[--violation]">{result.matches_found ?? 0}</p>
                <p className="text-[11px] text-muted-foreground">Matches</p>
              </div>
              <div className="rounded-md bg-muted/30 border border-border px-2 py-2">
                <p className="text-lg font-bold text-foreground">{result.total_violations_skipped ?? 0}</p>
                <p className="text-[11px] text-muted-foreground">Skipped</p>
              </div>
            </div>
          )}

          {matches.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                Ranked Results
              </p>
              {matches.map((m, i) => (
                <div
                  key={(m.violation?.timestamp ?? '') + i}
                  className={cn(
                    'flex items-center gap-3 rounded-md border px-3 py-2',
                    m.is_match ? 'bg-[--violation]/5 border-[--violation]/20' : 'bg-muted/20 border-border'
                  )}
                >
                  {m.violation?.annotated_image_path ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={evidenceImageUrl(m.violation.annotated_image_path)}
                      alt="Violation evidence"
                      className="size-12 rounded object-cover border border-border shrink-0"
                    />
                  ) : (
                    <div className="flex size-12 items-center justify-center rounded bg-muted border border-border shrink-0">
                      <ImageIcon className="size-4 text-muted-foreground" />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      {m.is_match ? (
                        <XCircle className="size-3.5 text-[--violation]" />
                      ) : (
                        <CheckCircle2 className="size-3.5 text-muted-foreground" />
                      )}
                      <span className="text-xs font-mono text-foreground truncate">
                        {m.violation?.timestamp ?? '—'}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 mt-0.5 text-[11px] text-muted-foreground">
                      {m.violation?.gps?.latitude != null && (
                        <span className="flex items-center gap-1">
                          <MapPin className="size-3" />
                          {m.violation.gps.latitude.toFixed(3)}, {m.violation.gps.longitude?.toFixed(3)}
                        </span>
                      )}
                      <span className="flex items-center gap-1">
                        <Clock className="size-3" />
                        {m.violation?.violations_count ?? 0} violation(s)
                      </span>
                    </div>
                  </div>
                  <Badge
                    variant="outline"
                    className={cn(
                      'text-[10px] h-5 px-1.5 font-mono shrink-0',
                      m.is_match ? 'text-[--violation] border-[--violation]/30' : 'text-muted-foreground'
                    )}
                  >
                    {(m.similarity * 100).toFixed(1)}%
                  </Badge>
                </div>
              ))}
            </div>
          )}

          {result.mode === 'violation_database_scan' && matches.length === 0 && (
            <p className="text-xs text-muted-foreground">No comparable violation images were found in the database.</p>
          )}
        </div>
      )}
    </div>
  )
}

// ── Page ────────────────────────────────────────────────────────────────────

export function FaceMatchPage() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2">
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <ScanFace className="size-4 text-primary" />
              Criminal Face Detection
            </CardTitle>
            <CardDescription className="text-muted-foreground text-sm">
              Match a reference face against another image, a stored violation, or the entire evidence database.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="compare" className="w-full">
              <TabsList className="mb-5 w-full justify-start bg-muted/40 border border-border h-9 p-0.5 rounded-lg">
                <TabsTrigger
                  value="compare"
                  className="flex items-center gap-1.5 text-xs h-8 px-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-md"
                >
                  <ScanFace className="size-3.5" />
                  Compare Two Images
                </TabsTrigger>
                <TabsTrigger
                  value="scan"
                  className="flex items-center gap-1.5 text-xs h-8 px-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-md"
                >
                  <Database className="size-3.5" />
                  Scan Violations
                </TabsTrigger>
              </TabsList>
              <TabsContent value="compare"><CompareMode /></TabsContent>
              <TabsContent value="scan"><ScanMode /></TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      {/* Info side panel */}
      <Card className="bg-card border-border h-fit">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <UserSearch className="size-4 text-primary" />
            How it works
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-xs text-muted-foreground">
          <div className="flex gap-2">
            <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-bold">1</span>
            <p><span className="text-foreground font-medium">Compare</span> — upload a reference face and a second image to check if they are the same person.</p>
          </div>
          <div className="flex gap-2">
            <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-[10px] font-bold">2</span>
            <p><span className="text-foreground font-medium">Scan</span> — match a reference face against one stored violation, or sweep the whole evidence database for hits.</p>
          </div>
          <Separator />
          <p className="leading-relaxed">
            Powered by DeepFace (Facenet512) on the backend. Faces are matched above the configured
            similarity threshold and ranked by score.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
