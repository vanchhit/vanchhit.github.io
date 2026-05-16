// Games section data. Add new entries here — see Guide.md.

export type Game = {
  id: string;
  title: string;
  description: string;
  url?: string;
  emoji: string;
  status: "playable" | "in-progress" | "concept";
};

export const GAMES: Game[] = [
  {
    id: "runway-rush",
    title: "Runway Rush",
    description: "Identify the airport from a single approach photo. Harder than it sounds.",
    emoji: "✈️",
    status: "concept",
  },
  {
    id: "metro-quiz",
    title: "Metro Map Quiz",
    description: "Name the city from a stylised slice of its transit map.",
    emoji: "🚇",
    status: "concept",
  },
  {
    id: "carbon-budget",
    title: "Carbon Budget",
    description: "Allocate a remaining gigaton budget across sectors. Watch the temperature dial.",
    emoji: "🌍",
    status: "concept",
  },
];
