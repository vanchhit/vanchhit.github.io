import { useState } from 'react';
import {
  ComposableMap,
  Geographies,
  Geography,
  ZoomableGroup,
} from 'react-simple-maps';
import { livedInCountries } from '../../data/lived-in-countries';

const GEO_URL =
  'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json';

const livedSet = new Set(livedInCountries.map((c) => c.id));
const countryMap = Object.fromEntries(
  livedInCountries.map((c) => [c.id, c])
);

interface Tooltip {
  name: string;
  note: string;
  x: number;
  y: number;
}

export default function WorldMap({ isDark }: { isDark?: boolean }) {
  const [tooltip, setTooltip] = useState<Tooltip | null>(null);

  const bgFill     = isDark ? '#1c1c1e' : '#f0f0f0';
  const defaultFill = isDark ? '#3a3a3c' : '#d2d2d7';
  const hoverFill  = isDark ? '#555558' : '#b0b0b8';
  const accentFill = isDark ? '#4db8ff' : '#0e5c8a';
  const strokeColor = isDark ? '#000' : '#fff';

  return (
    <div className="relative w-full rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-800" style={{ background: bgFill }}>
      <ComposableMap
        projectionConfig={{ scale: 147, center: [15, 10] }}
        width={800}
        height={420}
        style={{ width: '100%', height: 'auto' }}
      >
        <ZoomableGroup>
          <Geographies geography={GEO_URL}>
            {({ geographies }) =>
              geographies.map((geo) => {
                const id = geo.id as string;
                const lived = livedSet.has(id);
                const info = countryMap[id];

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={lived ? accentFill : defaultFill}
                    stroke={strokeColor}
                    strokeWidth={0.3}
                    style={{
                      default: { outline: 'none' },
                      hover:   { fill: lived ? accentFill : hoverFill, outline: 'none', cursor: lived ? 'pointer' : 'default' },
                      pressed: { outline: 'none' },
                    }}
                    onMouseEnter={(e) => {
                      if (!lived || !info) return;
                      const rect = (e.target as SVGElement)
                        .closest('svg')!
                        .getBoundingClientRect();
                      setTooltip({
                        name: info.name,
                        note: info.note,
                        x: e.clientX - rect.left,
                        y: e.clientY - rect.top,
                      });
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  />
                );
              })
            }
          </Geographies>
        </ZoomableGroup>
      </ComposableMap>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none px-3 py-2 rounded-lg text-sm shadow-lg"
          style={{
            left: tooltip.x + 12,
            top: tooltip.y - 10,
            background: isDark ? '#2c2c2e' : '#fff',
            color: isDark ? '#f5f5f7' : '#1d1d1f',
            border: `1px solid ${isDark ? '#3a3a3c' : '#d2d2d7'}`,
            maxWidth: 200,
          }}
        >
          <p className="font-bold">{tooltip.name}</p>
          <p style={{ color: isDark ? '#98989d' : '#6e6e73', fontSize: '0.75rem' }}>
            {tooltip.note}
          </p>
        </div>
      )}
    </div>
  );
}
