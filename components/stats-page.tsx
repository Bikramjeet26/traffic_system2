'use client'

import useSWR from 'swr'
import {
  BarChart3, ShieldAlert, CreditCard, TrendingUp, RefreshCw, AlertCircle, Car
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { fetchStats, violationLabel, SWR_KEYS } from '@/lib/api'

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ElementType
  label: string
  value: string | number
  sub?: string
  accent?: string
}) {
  return (
    <Card className="bg-card border-border">
      <CardContent className="pt-5 pb-4">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs text-muted-foreground uppercase tracking-widest">{label}</p>
            <p className={`text-3xl font-bold mt-1 ${accent ?? 'text-foreground'}`}>{value}</p>
            {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
          </div>
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted border border-border">
            <Icon className="size-4 text-muted-foreground" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function StatsPage() {
  const { data, error, isLoading, mutate } = useSWR(SWR_KEYS.stats, fetchStats, {
    refreshInterval: 60000,
  })

  const violationEntries = data && data.violation_counts
    ? Object.entries(data.violation_counts).sort((a, b) => b[1] - a[1])
    : []

  const vehicleEntries = data && data.vehicle_class_counts
    ? Object.entries(data.vehicle_class_counts).sort((a, b) => b[1] - a[1])
    : []

  const maxViolation = violationEntries[0]?.[1] ?? 1
  const maxVehicle = vehicleEntries[0]?.[1] ?? 1

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-foreground">System Statistics</h2>
          <p className="text-xs text-muted-foreground mt-0.5">Aggregated totals across all evidence records.</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => mutate()} className="gap-1.5">
          <RefreshCw className="size-3.5" />
          Refresh
        </Button>
      </div>

      <Separator />

      {/* Error */}
      {error && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="pt-4 flex items-start gap-3">
            <AlertCircle className="size-5 text-destructive shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-destructive">Failed to load stats</p>
              <p className="text-xs text-muted-foreground mt-1">{error.message}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Skeleton */}
      {isLoading && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="bg-card border-border">
              <CardContent className="pt-5 pb-4">
                <Skeleton className="h-4 w-24 mb-2" />
                <Skeleton className="h-8 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Data */}
      {data && (
        <>
          {/* KPI row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              icon={BarChart3}
              label="Total Records"
              value={data.total_records}
              sub="Evidence files on disk"
            />
            <StatCard
              icon={ShieldAlert}
              label="Total Violations"
              value={data.total_violations}
              accent="text-[--violation]"
              sub="Across all records"
            />
            <StatCard
              icon={CreditCard}
              label="Plates Read"
              value={data.total_plates}
              accent="text-[--plate]"
              sub="OCR detections"
            />
            <StatCard
              icon={TrendingUp}
              label="Avg Violations"
              value={(data.average_violations_per_record ?? 0).toFixed(2)}
              sub="Per record"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Violation breakdown */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <ShieldAlert className="size-4 text-[--violation]" />
                  Violation Breakdown
                </CardTitle>
                <CardDescription className="text-xs">
                  Counts by violation type
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                {violationEntries.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No violations recorded yet.</p>
                ) : (
                  violationEntries.map(([key, count]) => (
                    <div key={key} className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-foreground">{violationLabel(key)}</span>
                        <Badge variant="outline" className="text-[10px] h-4 px-1.5 font-mono text-[--violation] border-[--violation]/30">
                          {count}
                        </Badge>
                      </div>
                      <Progress
                        value={(count / maxViolation) * 100}
                        className="h-1.5 [&>div]:bg-[--violation]"
                      />
                    </div>
                  ))
                )}
              </CardContent>
            </Card>

            {/* Vehicle class breakdown */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Car className="size-4 text-primary" />
                  Vehicle Classes
                </CardTitle>
                <CardDescription className="text-xs">
                  Detected vehicle types from violations
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                {vehicleEntries.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No vehicle data recorded yet.</p>
                ) : (
                  vehicleEntries.map(([key, count]) => (
                    <div key={key} className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-foreground capitalize">{key}</span>
                        <Badge variant="outline" className="text-[10px] h-4 px-1.5 font-mono text-primary border-primary/30">
                          {count}
                        </Badge>
                      </div>
                      <Progress
                        value={(count / maxVehicle) * 100}
                        className="h-1.5 [&>div]:bg-primary"
                      />
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
