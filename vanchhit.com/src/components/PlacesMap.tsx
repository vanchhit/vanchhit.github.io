import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import type { Place } from "@/lib/travel";

// Fix Leaflet's default marker icon paths (they break under bundlers).
const icon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

export default function PlacesMap({ places }: { places: Place[] }) {
  // Defer mount one tick so SSR doesn't try to render Leaflet at all.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) {
    return <div className="flex h-[480px] items-center justify-center bg-muted text-sm text-muted-foreground">Loading map…</div>;
  }

  return (
    <MapContainer
      center={[20, 10]}
      zoom={2}
      style={{ height: "480px", width: "100%" }}
      scrollWheelZoom={false}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {places.map((p) => (
        <Marker key={p.id} position={[p.lat, p.lng]} icon={icon}>
          <Popup>
            <div className="text-sm">
              <div className="font-semibold">{p.name}</div>
              <div className="text-xs opacity-70">{p.country}</div>
              {p.note && <div className="mt-1">{p.note}</div>}
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
