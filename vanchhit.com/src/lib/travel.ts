// Travel data — countries visited (ISO-3 codes) + favourite places.
// Edit these to update the maps. See Guide.md for full instructions.

export type Place = {
  id: string;
  name: string;
  country: string;
  lat: number;
  lng: number;
  note?: string;
  photo?: string;
};

// Use ISO-3166-1 alpha-3 codes (e.g. "USA", "JPN", "DEU", "IND").
// A handy reference: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3
export const VISITED_COUNTRIES: string[] = [
  "IND", // India
  "USA",
  "GBR",
  "FRA",
  "DEU",
  "JPN",
  "SGP",
  "ARE",
];

export const FAVOURITE_PLACES: Place[] = [
  {
    id: "kyoto-philosophers-path",
    name: "Philosopher's Path",
    country: "Japan",
    lat: 35.0271,
    lng: 135.7944,
    note: "A quiet canal walk between temples — best in cherry-blossom season.",
  },
  {
    id: "sf-twin-peaks",
    name: "Twin Peaks",
    country: "USA",
    lat: 37.7544,
    lng: -122.4477,
    note: "The whole Bay laid out beneath you at sunset.",
  },
  {
    id: "lhr-myrtle-ave",
    name: "Myrtle Avenue, Heathrow",
    country: "United Kingdom",
    lat: 51.4587,
    lng: -0.4474,
    note: "Plane spotting under the 27L approach — heavies seem to skim the trees.",
  },
  {
    id: "delhi-lodhi",
    name: "Lodhi Gardens",
    country: "India",
    lat: 28.5933,
    lng: 77.2207,
    note: "Mughal-era tombs, slow joggers, parakeets at dusk.",
  },
];
