'use client'

import { useEffect, useRef, useState } from 'react'
import type { HeatmapPoint } from '@/lib/api'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'

interface HeatmapProps {
  points: HeatmapPoint[]
  isLoading?: boolean
}

export function Heatmap({ points, isLoading = false }: HeatmapProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMapRef = useRef<any>(null)
  const heatmapLayerRef = useRef<any>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || !mapRef.current || isLoading) return

    const initializeMap = async () => {
      try {
        // Import Leaflet CSS first
        await import('leaflet/dist/leaflet.css')
        const L = (await import('leaflet')).default

        // Initialize map if not already done
        if (!leafletMapRef.current) {
          // Default center (India)
          leafletMapRef.current = L.map(mapRef.current, {
            preferCanvas: true,
          }).setView([28.6139, 77.209], 12)

          // Add tile layer
          L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19,
          }).addTo(leafletMapRef.current)
        }

        // Convert points to heatmap format
        const validPoints = points.filter(p => !(p.lat === 0 && p.lon === 0))

        if (validPoints.length === 0) {
          return
        }

        // Remove old heatmap layer if exists
        if (heatmapLayerRef.current) {
          leafletMapRef.current.removeLayer(heatmapLayerRef.current)
        }

        // Create heatmap layer using simple colored circles
        const circleMarkers = L.featureGroup()
        const maxWeight = Math.max(...validPoints.map(p => p.weight), 1)

        validPoints.forEach((point) => {
          const intensity = Math.min(point.weight / maxWeight, 1)
          const radius = 3 + intensity * 12

          // Color gradient based on intensity
          let color = '#00ff00' // Green for low intensity
          if (intensity > 0.33) color = '#ffff00' // Yellow for medium
          if (intensity > 0.66) color = '#ff6600' // Orange for high
          if (intensity > 0.85) color = '#ff0000' // Red for very high

          const circle = L.circleMarker([point.lat, point.lon], {
            radius,
            fillColor: color,
            color: color,
            weight: 2,
            opacity: 0.8,
            fillOpacity: 0.5,
          })

          // Create popup with violation info
          const popupContent = `
            <div style="max-width: 200px; font-size: 12px;">
              <p><strong>Violations:</strong> ${point.violation_count}</p>
              <p><strong>Types:</strong> ${point.violation_types.join(', ') || 'N/A'}</p>
              <p><strong>Time:</strong> ${new Date(point.timestamp).toLocaleString()}</p>
              ${point.image_url ? `<img src="${point.image_url}" style="width: 100%; margin-top: 8px; border-radius: 4px; max-height: 150px;" />` : ''}
            </div>
          `
          circle.bindPopup(popupContent)
          circleMarkers.addLayer(circle)
        })

        circleMarkers.addTo(leafletMapRef.current)
        heatmapLayerRef.current = circleMarkers

        // Fit bounds to show all points
        if (validPoints.length > 0) {
          const bounds = L.latLngBounds(validPoints.map(d => [d.lat, d.lon]))
          leafletMapRef.current.fitBounds(bounds, { padding: [50, 50] })
        }
      } catch (err) {
        console.error('Error initializing map:', err)
      }
    }

    initializeMap()
  }, [points, isLoading, mounted])

  // Don't render until mounted on client
  if (!mounted) {
    return (
      <Card className="bg-card border-border overflow-hidden">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Violation Heatmap</CardTitle>
          <CardDescription className="text-xs text-muted-foreground">
            GPS distribution of traffic violations
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="h-[500px] w-full bg-muted flex items-center justify-center">
            <div className="text-muted-foreground text-sm">Loading map...</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-card border-border overflow-hidden">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Violation Heatmap</CardTitle>
        <CardDescription className="text-xs text-muted-foreground">
          GPS distribution of traffic violations. Circle size and color indicate violation frequency.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0 h-[500px]">
        <div
          ref={mapRef}
          className="h-full w-full bg-muted"
          style={{ position: 'relative' }}
        >
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10 pointer-events-none">
              <div className="text-white text-sm">Loading map data...</div>
            </div>
          )}
          {points.filter(p => !(p.lat === 0 && p.lon === 0)).length === 0 && !isLoading && (
            <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
              <div className="text-muted-foreground text-sm">No violation data to display</div>
            </div>
          )}
        </div>
        <div className="bg-muted/30 border-t border-border px-4 py-3 flex items-center gap-4 text-xs text-muted-foreground flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span>Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span>Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-orange-500"></div>
            <span>High</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span>Critical</span>
          </div>
          <span className="ml-auto">{points.filter(p => !(p.lat === 0 && p.lon === 0)).length} total locations</span>
        </div>
      </CardContent>
    </Card>
  )
}
