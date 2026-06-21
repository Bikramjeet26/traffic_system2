'use client'

import { Navbar } from '@/components/navbar'
import { AnalyzePage } from '@/components/analyze-page'
import { ViolationsPage } from '@/components/violations-page'
import { StatsPage } from '@/components/stats-page'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScanLine, ListFilter, BarChart3 } from 'lucide-react'
import { API_BASE } from '@/lib/api'

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />

      <main className="flex-1 container max-w-7xl mx-auto px-4 py-6">
        {/* Page header */}
        <div className="mb-6">
          <h1 className="text-xl font-semibold text-foreground text-balance">
            Traffic Violation Detection
          </h1>
          <p className="text-sm text-muted-foreground mt-1 text-pretty">
            Analyze road images with AI — detect vehicles, read license plates, and flag helmet &amp; seatbelt violations.
          </p>
        </div>

        <Tabs defaultValue="analyze" className="w-full">
          <TabsList className="mb-6 w-full justify-start bg-card border border-border h-9 p-0.5 rounded-lg">
            <TabsTrigger
              value="analyze"
              className="flex items-center gap-1.5 text-xs h-8 px-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-md"
            >
              <ScanLine className="size-3.5" />
              Analyze Image
            </TabsTrigger>
            <TabsTrigger
              value="violations"
              className="flex items-center gap-1.5 text-xs h-8 px-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-md"
            >
              <ListFilter className="size-3.5" />
              Violations Log
            </TabsTrigger>
            <TabsTrigger
              value="stats"
              className="flex items-center gap-1.5 text-xs h-8 px-3 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground rounded-md"
            >
              <BarChart3 className="size-3.5" />
              Statistics
            </TabsTrigger>
          </TabsList>

          <TabsContent value="analyze">
            <AnalyzePage />
          </TabsContent>

          <TabsContent value="violations">
            <ViolationsPage />
          </TabsContent>

          <TabsContent value="stats">
            <StatsPage />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-4 px-4 mt-auto">
        <div className="container max-w-7xl mx-auto flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            TrafficVision — Powered by YOLOv11 &amp; RapidOCR
          </p>
          <p className="text-xs text-muted-foreground">
            Backend:{' '}
            <a
              href={`${API_BASE}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              HF Spaces API Docs
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}
