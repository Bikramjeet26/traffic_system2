'use client'

import { useState } from 'react'
import useSWR from 'swr'
import { Clock, MapPin, CreditCard, ShieldAlert, Image as ImageIcon, ChevronDown, ChevronUp, RefreshCw, AlertCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Separator } from '@/components/ui/separator'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow
} from '@/components/ui/table'
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription
} from '@/components/ui/dialog'
import { fetchViolations, evidenceImageUrl, violationLabel, SWR_KEYS, type EvidenceRecord } from '@/lib/api'
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

function EvidenceRow({ record }: { record: EvidenceRecord }) {
  const [open, setOpen] = useState(false)
  const [imgOpen, setImgOpen] = useState(false)

  const hasViolations = record.violations.length > 0
  const imgSrc = evidenceImageUrl(record.annotated_image_path)

  return (
    <>
      <TableRow
        className={cn(
          'cursor-pointer transition-colors hover:bg-muted/20',
          hasViolations && 'border-l-2 border-l-[--violation]'
        )}
        onClick={() => setOpen(o => !o)}
      >
        {/* Timestamp */}
        <TableCell className="font-mono text-xs text-muted-foreground whitespace-nowrap">
          {new Date(record.timestamp).toLocaleString()}
        </TableCell>

        {/* Violations */}
        <TableCell>
          {hasViolations ? (
            <div className="flex flex-wrap gap-1">
              {record.violations.map((v, i) => <ViolationTypeBadge key={i} type={v.type} />)}
            </div>
          ) : (
            <Badge variant="outline" className="text-[11px] text-[--safe] border-[--safe]/30">Clean</Badge>
          )}
        </TableCell>

        {/* Plates */}
        <TableCell>
          {record.plates.length > 0 ? (
            <div className="flex flex-wrap gap-1">
              {record.plates.map((p, i) => <PlateChip key={i} text={p.text} />)}
            </div>
          ) : (
            <span className="text-xs text-muted-foreground">None</span>
          )}
        </TableCell>

        {/* GPS */}
        <TableCell className="text-xs text-muted-foreground font-mono">
          {record.gps[0] !== 0 || record.gps[1] !== 0
            ? `${record.gps[0].toFixed(4)}, ${record.gps[1].toFixed(4)}`
            : <span className="italic">—</span>
          }
        </TableCell>

        {/* Actions */}
        <TableCell className="text-right">
          <div className="flex items-center justify-end gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="size-7 p-0"
              onClick={e => { e.stopPropagation(); setImgOpen(true) }}
              aria-label="View annotated image"
            >
              <ImageIcon className="size-3.5" />
            </Button>
            {open ? <ChevronUp className="size-4 text-muted-foreground" /> : <ChevronDown className="size-4 text-muted-foreground" />}
          </div>
        </TableCell>
      </TableRow>

      {/* Expanded detail row */}
      {open && (
        <TableRow className="bg-muted/10 hover:bg-muted/10">
          <TableCell colSpan={5} className="py-3 px-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Violations detail */}
              {record.violations.length > 0 && (
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">Violation Details</p>
                  <div className="flex flex-col gap-1.5">
                    {record.violations.map((v, i) => (
                      <div key={i} className="flex items-center justify-between rounded bg-[--violation]/5 border border-[--violation]/10 px-2.5 py-1.5 text-xs">
                        <span className="text-[--violation] font-medium">
                          {violationLabel(v.type)}
                        </span>
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

              {/* Plates detail */}
              {record.plates.length > 0 && (
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">License Plates</p>
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

      {/* Image dialog */}
      <Dialog open={imgOpen} onOpenChange={setImgOpen}>
        <DialogContent className="max-w-3xl bg-card border-border">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-sm">
              <ImageIcon className="size-4 text-primary" />
              Annotated Evidence Image
            </DialogTitle>
            <DialogDescription className="text-xs font-mono text-muted-foreground">
              {record.timestamp}
            </DialogDescription>
          </DialogHeader>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imgSrc}
            alt="Annotated traffic scene"
            className="w-full rounded-lg object-contain max-h-[70vh]"
          />
          <div className="flex items-center gap-4 text-xs text-muted-foreground pt-1">
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-full inline-block bg-[--safe]" />
              Vehicles (green)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-full inline-block bg-[--plate]" />
              Plates (yellow)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-full inline-block bg-[--violation]" />
              Violations (red)
            </span>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}

export function ViolationsPage() {
  const { data, error, isLoading, mutate } = useSWR(SWR_KEYS.violations, fetchViolations, {
    refreshInterval: 60000,
  })

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Evidence Records</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            All stored detection records — newest first. Click a row to expand details.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => mutate()} className="gap-1.5">
          <RefreshCw className="size-3.5" />
          Refresh
        </Button>
      </div>

      <Separator />

      {/* Error state */}
      {error && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="pt-4 flex items-start gap-3">
            <AlertCircle className="size-5 text-destructive shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-destructive">Failed to load violations</p>
              <p className="text-xs text-muted-foreground mt-1">{error.message}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Loading skeleton */}
      {isLoading && (
        <Card className="bg-card border-border">
          <CardContent className="pt-4 flex flex-col gap-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center gap-3">
                <Skeleton className="h-4 w-36" />
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 w-20" />
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Empty state */}
      {!isLoading && !error && data && data.length === 0 && (
        <Card className="bg-card border-border border-dashed min-h-[200px] flex items-center justify-center">
          <div className="flex flex-col items-center gap-3 text-center px-8">
            <div className="flex size-12 items-center justify-center rounded-full bg-muted border border-border">
              <ShieldAlert className="size-6 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">No violation records yet. Analyze an image to create evidence records.</p>
          </div>
        </Card>
      )}

      {/* Data table */}
      {!isLoading && !error && data && data.length > 0 && (
        <Card className="bg-card border-border overflow-hidden">
          <CardHeader className="pb-2 pt-4 px-4">
            <CardTitle className="text-sm flex items-center gap-2">
              <ShieldAlert className="size-4 text-primary" />
              {data.length} Record{data.length !== 1 ? 's' : ''} Found
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="border-border hover:bg-transparent">
                    <TableHead className="text-xs">
                      <span className="flex items-center gap-1.5"><Clock className="size-3" /> Timestamp</span>
                    </TableHead>
                    <TableHead className="text-xs">
                      <span className="flex items-center gap-1.5"><ShieldAlert className="size-3" /> Violations</span>
                    </TableHead>
                    <TableHead className="text-xs">
                      <span className="flex items-center gap-1.5"><CreditCard className="size-3" /> Plates</span>
                    </TableHead>
                    <TableHead className="text-xs">
                      <span className="flex items-center gap-1.5"><MapPin className="size-3" /> GPS</span>
                    </TableHead>
                    <TableHead className="text-xs text-right">Evidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.map((record, i) => (
                    <EvidenceRow key={record.timestamp + i} record={record} />
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
