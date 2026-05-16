// Single source of truth for blog topics + persona metadata.
// To add a new topic: append here, create a route file src/routes/blog.<slug>.tsx
// from the existing topic route template, and use the slug in post frontmatter.

export type TopicSlug = "ai-safety" | "climate" | "aviation" | "trains" | "general";

export type Topic = {
  slug: TopicSlug;
  label: string;
  short: string;
  description: string;
  motif: "grid" | "blueprint" | "topo" | "none";
};

export const TOPICS: Topic[] = [
  {
    slug: "aviation",
    label: "Aviation & Plane Spotting",
    short: "Aviation",
    description: "Aircraft, airports, flight ops, and time spent at the fence.",
    motif: "blueprint",
  },
  {
    slug: "trains",
    label: "Trains & Transportation",
    short: "Trains",
    description: "Rail networks, transit, infrastructure, and how cities move.",
    motif: "none",
  },
  {
    slug: "ai-safety",
    label: "AI Safety",
    short: "AI Safety",
    description: "Notes on alignment, evaluations, governance, and risk.",
    motif: "grid",
  },
  {
    slug: "climate",
    label: "Climate Change",
    short: "Climate",
    description: "Decarbonisation, adaptation, and the systems behind both.",
    motif: "topo",
  },
  {
    slug: "general",
    label: "General",
    short: "General",
    description: "Everything else worth writing down.",
    motif: "none",
  },
];

export const TOPIC_BY_SLUG: Record<TopicSlug, Topic> = Object.fromEntries(
  TOPICS.map((t) => [t.slug, t]),
) as Record<TopicSlug, Topic>;

export function topicLabel(slug: string): string {
  return TOPIC_BY_SLUG[slug as TopicSlug]?.label ?? "General";
}

export function topicMotifClass(slug: string): string {
  const m = TOPIC_BY_SLUG[slug as TopicSlug]?.motif ?? "none";
  return m === "grid" ? "motif-grid" : m === "blueprint" ? "motif-blueprint" : m === "topo" ? "motif-topo" : "";
}
