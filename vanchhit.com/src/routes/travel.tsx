import { createFileRoute } from "@tanstack/react-router";
import { lazy, Suspense } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps";
import countries from "i18n-iso-countries";
import enLocale from "i18n-iso-countries/langs/en.json";
import { VISITED_COUNTRIES, FAVOURITE_PLACES } from "@/lib/travel";

countries.registerLocale(enLocale);

// World atlas (TopoJSON, ISO-3166-1 numeric IDs as strings)
const GEO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

// Leaflet uses `window`. Lazy + Suspense keeps it client-only and avoids SSR breakage.
const PlacesMap = lazy(() => import("@/components/PlacesMap"));

export const Route = createFileRoute("/travel")({
  head: () => ({
    meta: [
      { title: "Travel — Vanchhit" },
      {
        name: "description",
        content: "An interactive map of countries I've visited and a few favourite places worth a detour.",
      },
      { property: "og:title", content: "Travel — Vanchhit" },
      { property: "og:description", content: "Countries visited and favourite places." },
    ],
  }),
  component: TravelPage,
});

function TravelPage() {
  const visitedNumeric = new Set(
    VISITED_COUNTRIES.map((alpha3) => countries.alpha3ToNumeric(alpha3)).filter(Boolean) as string[],
  );

  return (
    <div className="mx-auto max-w-6xl px-5 py-20">
      <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">Travel</div>
      <h1 className="mt-3 font-display text-5xl font-semibold leading-tight tracking-tight">
        {VISITED_COUNTRIES.length} countries.
      </h1>
      <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
        Below: a quiet world map of where I've been, and an interactive map of
        places I'd happily go back to.
      </p>

      {/* World countries map */}
      <section className="mt-14 overflow-hidden rounded-2xl border border-border/60 bg-muted/30">
        <ComposableMap
          projection="geoEqualEarth"
          projectionConfig={{ scale: 155 }}
          style={{ width: "100%", height: "auto" }}
        >
          <Geographies geography={GEO_URL}>
            {({ geographies }: { geographies: Array<{ rsmKey: string; id: string; properties: { name: string } }> }) =>
              geographies.map((geo) => {
                const visited = visitedNumeric.has(geo.id);
                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    style={{
                      default: {
                        fill: visited ? "var(--accent)" : "var(--muted)",
                        stroke: "var(--border)",
                        strokeWidth: 0.5,
                        outline: "none",
                      },
                      hover: {
                        fill: visited ? "var(--accent)" : "var(--secondary)",
                        outline: "none",
                      },
                      pressed: { outline: "none" },
                    }}
                  >
                    <title>{geo.properties.name}{visited ? " · visited" : ""}</title>
                  </Geography>
                );
              })
            }
          </Geographies>
        </ComposableMap>
        <div className="flex items-center gap-4 border-t border-border/60 px-5 py-3 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          <span className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: "var(--accent)" }} />
            Visited
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-sm bg-muted-foreground/30" />
            Not yet
          </span>
        </div>
      </section>

      {/* Favourite places */}
      <section className="mt-20">
        <div className="flex items-end justify-between">
          <h2 className="font-display text-3xl font-semibold tracking-tight">Favourite places</h2>
          <div className="font-mono text-xs text-muted-foreground">{FAVOURITE_PLACES.length} pins</div>
        </div>
        <p className="mt-3 max-w-2xl text-muted-foreground">
          Pin and a sentence each. Click a pin for the note.
        </p>

        <div className="mt-8 overflow-hidden rounded-2xl border border-border/60">
          <Suspense fallback={<div className="flex h-[480px] items-center justify-center bg-muted text-sm text-muted-foreground">Loading map…</div>}>
            <PlacesMap places={FAVOURITE_PLACES} />
          </Suspense>
        </div>

        <ul className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2">
          {FAVOURITE_PLACES.map((p) => (
            <li key={p.id} className="rounded-xl border border-border/60 bg-card p-4">
              <div className="font-display text-base font-semibold">{p.name}</div>
              <div className="mt-0.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                {p.country}
              </div>
              {p.note && <div className="mt-2 text-sm text-muted-foreground">{p.note}</div>}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
