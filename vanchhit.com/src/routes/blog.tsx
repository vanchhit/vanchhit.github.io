import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import Fuse from "fuse.js";
import { BLOG_POSTS } from "@/lib/content";
import { TOPICS, topicLabel, type TopicSlug } from "@/lib/topics";
import { Search } from "lucide-react";

export const Route = createFileRoute("/blog")({
  head: () => ({
    meta: [
      { title: "Blog — Vanchhit" },
      { name: "description", content: "All writing — aviation, trains, AI safety, climate, and more." },
      { property: "og:title", content: "Blog — Vanchhit" },
      { property: "og:description", content: "All writing." },
    ],
  }),
  component: BlogIndex,
});

function BlogIndex() {
  const [topic, setTopic] = useState<TopicSlug | "all">("all");
  const [query, setQuery] = useState("");

  const fuse = useMemo(
    () => new Fuse(BLOG_POSTS, { keys: ["title", "excerpt", "tags", "topic"], threshold: 0.35 }),
    [],
  );

  const filtered = useMemo(() => {
    let posts = BLOG_POSTS;
    if (topic !== "all") posts = posts.filter((p) => p.topic === topic);
    if (query.trim()) {
      const matches = fuse.search(query.trim()).map((r) => r.item);
      const set = new Set(matches.map((m) => m.slug));
      posts = posts.filter((p) => set.has(p.slug));
    }
    return posts;
  }, [topic, query, fuse]);

  return (
    <div className="mx-auto max-w-6xl px-5 py-20">
      <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">Blog</div>
      <h1 className="mt-3 font-display text-5xl font-semibold leading-tight tracking-tight">
        Writing.
      </h1>
      <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
        Notes on the things I keep coming back to. Filter by topic, or search.
      </p>

      <div className="mt-10 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setTopic("all")}
            className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
              topic === "all" ? "border-foreground bg-foreground text-background" : "border-border text-muted-foreground hover:text-foreground"
            }`}
          >
            All
          </button>
          {TOPICS.filter((t) => t.slug !== "general").map((t) => (
            <button
              key={t.slug}
              data-topic={t.slug}
              onClick={() => setTopic(t.slug)}
              className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                topic === t.slug
                  ? "border-transparent text-background"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
              style={topic === t.slug ? { background: "var(--accent)" } : undefined}
            >
              {t.short}
            </button>
          ))}
        </div>

        <div className="relative w-full sm:w-72">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search posts…"
            className="w-full rounded-full border border-input bg-background py-2 pl-9 pr-3 text-sm outline-none focus:border-accent"
          />
        </div>
      </div>

      <div className="mt-10 divide-y divide-border/60">
        {filtered.length === 0 && (
          <div className="py-10 text-sm text-muted-foreground">No posts match.</div>
        )}
        {filtered.map((p) => (
          <Link
            key={p.slug}
            to="/blog/$slug"
            params={{ slug: p.slug }}
            data-topic={p.topic}
            className="group flex flex-col gap-2 py-6 transition-colors sm:grid sm:grid-cols-[120px_1fr_120px] sm:items-baseline sm:gap-6"
          >
            <div className="font-mono text-xs text-muted-foreground">{p.date}</div>
            <div>
              <div className="font-display text-xl font-semibold tracking-tight transition-colors group-hover:text-accent">
                {p.title}
              </div>
              <div className="mt-1 line-clamp-2 text-sm text-muted-foreground">{p.excerpt}</div>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-widest sm:text-right" style={{ color: "var(--accent)" }}>
              {topicLabel(p.topic)}
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
