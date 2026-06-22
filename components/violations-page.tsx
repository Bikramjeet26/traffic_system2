'use client'

import { useState, useMemo } from 'react'
import useSWR from 'swr'
import { Clock, MapPin, CreditCard, ShieldAlert, Image as ImageIcon, ChevronDown, ChevronUp, RefreshCw, AlertCircle, CheckCircle2, Save, Users } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow
} from '@/components/ui/table'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription
} from '@/components/ui/dialog'
import {
  fetchCollectionViolations,
  fetchCollectionViolationDetail,
  verifyViolation,
  evidenceAssetUrl,
  violationLabel,
  type CollectionViolationRecord,
  type CollectionViolationDetailResponse,
  type ModelVerificationLabel,
} from '@/lib/api'
import { cn } from '@/lib/utils'

function ViolationTypeBadge({ type }: { type: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium bg-[--violation]/10 text-[--violation] border border-[--violation]/20">
      <ShieldAlert className="size-2.5" />
      {violationLabel(type)}
    </span>
  )
}

function PlateChip({ text }: { text: string }) {
  return (
    <span className="font-mono text-[11px] font-semibold bg-[--plate]/10 text-[--plate] border border-[--plate]/20 rounded px-1.5 py-0.5">
      {text || '—'}
    </span>
  )
}

// ── Violations Query Part ──
function ViolationsQueryPart() {
  const { data: records, isLoading, error, mutate } = useSWR(
    'collection-violations',
    () => fetchCollectionViolations(true),
    { revalidateOnFocus: false }
  )

  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [imgOpen, setImgOpen] = useState(false)
  const [selectedRecord, setSelectedRecord] = useState<CollectionViolationRecord | null>(null)

  const handleImageOpen = (record: CollectionViolationRecord) => {
    setSelectedRecord(record)
    setImgOpen(true)
  }

  if (error) {
    return (
      <Card className="border-destructive/30 bg-destructive/5">
        <CardContent className="pt-4 flex items-start gap-3">
          <AlertCircle className="size-5 text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-destructive">Failed to load violations</p>
            <p className="text-xs text-muted-foreground mt-1">{error.message}</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-foreground">
            {isLoading ? 'Loading...' : `${records?.length ?? 0} violations found`}
          </p>
          <p className="text-xs text-muted-foreground">Complete log of all detected traffic violations</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5"
          onClick={() => mutate()}
          disabled={isLoading}
        >
          <RefreshCw className={cn('size-3.5', isLoading && 'animate-spin')} />
          Refresh
        </Button>
      </div>

      <Card className="bg-card border-border overflow-hidden">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-border">
                <TableHead className="text-xs">Time</TableHead>
                <TableHead className="text-xs">Violations</TableHead>
                <TableHead className="text-xs">Plates</TableHead>
                <TableHead className="text-xs">Location</TableHead>
                <TableHead className="text-xs text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array(3).fill(0).map((_, i) => (
                  <TableRow key={i}>
                    <TableCell colSpan={5} className="py-3"><Skeleton className="h-4" /></TableCell>
                  </TableRow>
                ))
              ) : records && records.length > 0 ? (
                records.map((record) => {
                  const isExpanded = expandedId === record.id
                  const hasViolations = record.violations.length > 0

                  return (
                    <div key={record.id}>
                      <TableRow
                        className={cn(
                          'cursor-pointer transition-colors hover:bg-muted/20',
                          hasViolations && 'border-l-2 border-l-[--violation]'
                        )}
                        onClick={() => setExpandedId(isExpanded ? null : record.id)}
                      >
                        <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
                          {new Date(record.timestamp).toLocaleString()}
                        </TableCell>

                        <TableCell>
                          {hasViolations ? (
                            <div className="flex flex-wrap gap-1">
                              {record.violations.slice(0, 3).map((v, i) => <ViolationTypeBadge key={i} type={v.type} />)}
                              {record.violations.length > 3 && (
                                <Badge variant="outline" className="text-[10px]">+{record.violations.length - 3}</Badge>
                              )}
                            </div>
                          ) : (
                            <Badge variant="outline" className="text-[11px] text-[--safe] border-[--safe]/30">Clean</Badge>
                          )}
                        </TableCell>

                        <TableCell>
                          {record.plates.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {record.plates.slice(0, 2).map((p, i) => <PlateChip key={i} text={p.text} />)}
                              {record.plates.length > 2 && (
                                <span className="text-xs text-muted-foreground">+{record.plates.length - 2}</span>
                              )}
                            </div>
                          ) : (
                            <span className="text-xs text-muted-foreground">None</span>
                          )}
                        </TableCell>

                        <TableCell className="text-xs text-muted-foreground font-mono">
                          {record.gps_lat !== 0 || record.gps_lon !== 0
                            ? `${record.gps_lat.toFixed(4)}, ${record.gps_lon.toFixed(4)}`
                            : <span className="italic">—</span>
                          }
                        </TableCell>

                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="size-7 p-0"
                              onClick={e => { e.stopPropagation(); handleImageOpen(record) }}
                              aria-label="View annotated image"
                            >
                              <ImageIcon className="size-3.5" />
                            </Button>
                            {isExpanded ? <ChevronUp className="size-4 text-muted-foreground" /> : <ChevronDown className="size-4 text-muted-foreground" />}
                          </div>
                        </TableCell>
                      </TableRow>

                      {isExpanded && (
                        <TableRow className="bg-muted/10 hover:bg-muted/10">
                          <TableCell colSpan={5} className="py-3 px-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {record.violations.length > 0 && (
                                <div>
                                  <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">All Violations</p>
                                  <div className="flex flex-col gap-1.5">
                                    {record.violations.map((v, i) => (
                                      <div key={i} className="flex items-center justify-between rounded bg-[--violation]/5 border border-[--violation]/10 px-2.5 py-1.5 text-xs">
                                        <span className="text-[--violation] font-medium">{violationLabel(v.type)}</span>
                                        <div className="flex items-center gap-3 text-muted-foreground">
                                          <span>Vehicle: {v.vehicle_class}</span>
                                          <Badge variant="outline" className="text-[10px] h-4 px-1 font-mono">
                                            {(v.confidence * 100).toFixed(0)}%
                                          </Badge>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {record.plates.length > 0 && (
                                <div>
                                  <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">All License Plates</p>
                                  <div className="flex flex-col gap-1.5">
                                    {record.plates.map((p, i) => (
                                      <div key={i} className="flex items-center justify-between rounded bg-[--plate]/5 border border-[--plate]/10 px-2.5 py-1.5 text-xs">
                                        <span className="font-mono font-semibold text-[--plate]">{p.text || '(unread)'}</span>
                                        <Badge variant="outline" className="text-[10px] h-4 px-1 font-mono">
                                          {(p.confidence * 100).toFixed(0)}%
                                        </Badge>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      )}
                    </div>
                  )
                })
              ) : (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                    No violations found in collection
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </Card>

      {/* Image Modal */}
      <Dialog open={imgOpen} onOpenChange={setImgOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Annotated Evidence Image</DialogTitle>
            <DialogDescription>
              {selectedRecord && new Date(selectedRecord.timestamp).toLocaleString()}
            </DialogDescription>
          </DialogHeader>
          {selectedRecord && (
            <div className="relative w-full">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={evidenceAssetUrl(selectedRecord.image_url) || "/placeholder.svg"}
                alt="Annotated evidence"
                className="w-full rounded-lg"
              />
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

// ── Violation Verification Part ──
function ViolationVerificationPart() {
  const { data: records, isLoading, mutate } = useSWR(
    'collection-violations-verify',
    () => fetchCollectionViolations(true),
    { revalidateOnFocus: false }
  )

  const [selectedViolationId, setSelectedViolationId] = useState<string | null>(null)
  const [verificationData, setVerificationData] = useState<CollectionViolationDetailResponse | null>(null)
  const [verifyLoading, setVerifyLoading] = useState(false)
  const [verifyError, setVerifyError] = useState<string | null>(null)

  // Verification form state
  const [violationConfirmed, setViolationConfirmed] = useState(false)
  const [labels, setLabels] = useState({
    ocr: { detected: false, correct: false, notes: '' },
    license_plate: { detected: false, correct: false, notes: '' },
    vehicle: { detected: false, correct: false, notes: '' },
    helmet: { detected: false, correct: false, notes: '' },
    seatbelt: { detected: false, correct: false, notes: '' },
  })
  const [annotationNotes, setAnnotationNotes] = useState('')

  const handleSelectViolation = async (violationId: string) => {
    setSelectedViolationId(violationId)
    setVerifyError(null)
    try {
      const detail = await fetchCollectionViolationDetail(violationId)
      setVerificationData(detail)
    } catch (err) {
      setVerifyError(err instanceof Error ? err.message : 'Failed to load violation detail')
    }
  }

  const handleSubmitVerification = async () => {
    if (!selectedViolationId || !verificationData) return

    setVerifyLoading(true)
    setVerifyError(null)

    try {
      const request = {
        violation_id: selectedViolationId,
        violation_confirmed: violationConfirmed,
        ocr: labels.ocr as ModelVerificationLabel,
        license_plate: labels.license_plate as ModelVerificationLabel,
        vehicle: labels.vehicle as ModelVerificationLabel,
        helmet: labels.helmet as ModelVerificationLabel,
        seatbelt: labels.seatbelt as ModelVerificationLabel,
        annotation_notes: annotationNotes || undefined,
      }

      await verifyViolation(request)

      // Reset form
      setSelectedViolationId(null)
      setVerificationData(null)
      setViolationConfirmed(false)
      setLabels({
        ocr: { detected: false, correct: false, notes: '' },
        license_plate: { detected: false, correct: false, notes: '' },
        vehicle: { detected: false, correct: false, notes: '' },
        helmet: { detected: false, correct: false, notes: '' },
        seatbelt: { detected: false, correct: false, notes: '' },
      })
      setAnnotationNotes('')
      mutate()
    } catch (err) {
      setVerifyError(err instanceof Error ? err.message : 'Verification failed')
    } finally {
      setVerifyLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <p className="text-sm font-medium text-foreground">Validate Model Output</p>
        <p className="text-xs text-muted-foreground">Review and correct model predictions for training dataset improvement</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Violations list */}
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground px-1">Select Violation</p>
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {isLoading ? (
              Array(3).fill(0).map((_, i) => <Skeleton key={i} className="h-16" />)
            ) : records && records.length > 0 ? (
              records.map((record) => (
                <button
                  key={record.id}
                  onClick={() => handleSelectViolation(record.id)}
                  className={cn(
                    'w-full text-left p-3 rounded-lg border transition-colors',
                    selectedViolationId === record.id
                      ? 'bg-primary/10 border-primary'
                      : 'border-border hover:bg-muted/50'
                  )}
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="font-mono text-xs text-muted-foreground">
                      {new Date(record.timestamp).toLocaleString()}
                    </span>
                    {record.violations.length > 0 && (
                      <Badge variant="outline" className="text-[10px]">{record.violations.length}</Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    {record.violations.length > 0 && (
                      <>
                        <ShieldAlert className="size-3 text-[--violation]" />
                        <span className="text-[--violation] font-medium">
                          {violationLabel(record.violations[0].type)}
                        </span>
                      </>
                    )}
                  </div>
                </button>
              ))
            ) : (
              <p className="text-xs text-muted-foreground text-center py-8">No violations to verify</p>
            )}
          </div>
        </div>

        {/* Verification form */}
        <div className="space-y-4">
          {verifyError && (
            <Card className="border-destructive/30 bg-destructive/5">
              <CardContent className="pt-3">
                <p className="text-xs text-destructive">{verifyError}</p>
              </CardContent>
            </Card>
          )}

          {selectedViolationId && verificationData ? (
            <>
              {/* Evidence image */}
              <Card className="bg-card border-border overflow-hidden">
                <CardContent className="p-4 flex items-center justify-center bg-muted/20">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={evidenceAssetUrl(verificationData.record.image_url) || "/placeholder.svg"}
                    alt="Evidence"
                    className="max-w-full max-h-[600px] object-contain"
                  />
                </CardContent>
              </Card>

              {/* Model outputs */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Model Detections</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex items-center justify-between text-xs p-2 bg-muted/30 rounded">
                    <span>Vehicle Detected:</span>
                    <Badge variant={verificationData.record.system_outputs.vehicle_detected ? 'default' : 'outline'}>
                      {verificationData.record.system_outputs.vehicle_detected ? 'Yes' : 'No'}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-xs p-2 bg-muted/30 rounded">
                    <span>Plate Detected:</span>
                    <Badge variant={verificationData.record.system_outputs.license_plate_detected ? 'default' : 'outline'}>
                      {verificationData.record.system_outputs.license_plate_detected ? 'Yes' : 'No'}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-xs p-2 bg-muted/30 rounded">
                    <span>OCR Detected:</span>
                    <Badge variant={verificationData.record.system_outputs.ocr_detected ? 'default' : 'outline'}>
                      {verificationData.record.system_outputs.ocr_detected ? 'Yes' : 'No'}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* Verification Form */}
              <Card className="bg-card border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Your Verification</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {/* Violation confirmed */}
                  <div className="flex items-center gap-3 p-2 bg-muted/20 rounded">
                    <label className="flex items-center gap-2 cursor-pointer flex-1">
                      <input
                        type="checkbox"
                        checked={violationConfirmed}
                        onChange={e => setViolationConfirmed(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <span className="text-xs font-medium">Violation Confirmed</span>
                    </label>
                  </div>

                  {/* Notes */}
                  <div>
                    <p className="text-xs font-medium mb-2">Annotation Notes</p>
                    <textarea
                      value={annotationNotes}
                      onChange={e => setAnnotationNotes(e.target.value)}
                      placeholder="Any corrections or notes..."
                      className="w-full text-xs p-2 rounded border border-border bg-background"
                      rows={3}
                    />
                  </div>

                  {/* Submit */}
                  <Button
                    className="w-full gap-2"
                    onClick={handleSubmitVerification}
                    disabled={verifyLoading}
                  >
                    <Save className="size-3.5" />
                    Submit Verification
                  </Button>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="bg-muted/30 border-border border-dashed flex items-center justify-center min-h-[400px]">
              <div className="text-center">
                <ShieldAlert className="size-8 text-muted-foreground mx-auto mb-2" />
                <p className="text-sm text-muted-foreground">Select a violation to review</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

export function ViolationsPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Violations &amp; Verification</h1>
        <p className="text-sm text-muted-foreground mt-1">Review detected violations and validate model predictions for continuous improvement</p>
      </div>

      <Tabs defaultValue="query" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="query" className="gap-2">
            <ShieldAlert className="size-3.5" />
            <span className="hidden sm:inline">Violations Query</span>
            <span className="sm:hidden">Query</span>
          </TabsTrigger>
          <TabsTrigger value="verify" className="gap-2">
            <CheckCircle2 className="size-3.5" />
            <span className="hidden sm:inline">Verification</span>
            <span className="sm:hidden">Verify</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="query" className="space-y-4">
          <ViolationsQueryPart />
        </TabsContent>

        <TabsContent value="verify" className="space-y-4">
          <ViolationVerificationPart />
        </TabsContent>
      </Tabs>
    </div>
  )
}
