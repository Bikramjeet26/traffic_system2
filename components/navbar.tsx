'use client'

import { Activity, Shield } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import useSWR from 'swr'
import { fetchHealth, SWR_KEYS } from '@/lib/api'

export function Navbar() {
  const { data: health } = useSWR(SWR_KEYS.health, fetchHealth, {
    refreshInterval: 30000,
  })

  const isOnline = health?.status === 'ok' || health?.status === 'healthy'

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-card/90 backdrop-blur-sm">
      <div className="flex h-14 items-center px-4 gap-3">
        {/* Brand */}
        <div className="flex items-center gap-2">
          <div className="flex size-8 items-center justify-center rounded bg-primary/10 border border-primary/20">
            <Shield className="size-4 text-primary" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="font-semibold text-foreground text-sm tracking-tight">TrafficVision</span>
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest">Enforcement System</span>
          </div>
        </div>

        <Separator orientation="vertical" className="h-6 mx-2" />

        {/* Nav breadcrumb hint */}
        <span className="text-xs text-muted-foreground hidden sm:block">
          AI-Powered Violation Detection
        </span>

        <div className="ml-auto flex items-center gap-3">
          {/* Backend status */}
          <div className="flex items-center gap-1.5">
            <Activity className="size-3.5 text-muted-foreground" />
            <span className="text-xs text-muted-foreground hidden sm:block">HF Space</span>
            {health ? (
              <Badge
                variant={isOnline ? 'default' : 'destructive'}
                className="text-[10px] h-4 px-1.5"
              >
                {isOnline ? 'Online' : 'Offline'}
              </Badge>
            ) : (
              <Badge variant="secondary" className="text-[10px] h-4 px-1.5">
                Checking…
              </Badge>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
