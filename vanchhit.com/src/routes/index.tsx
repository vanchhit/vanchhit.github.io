import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpRight } from "lucide-react";
import { BLOG_POSTS } from "@/lib/content";
import { TOPICS, topicLabel } from "@/lib/topics";
import { VISITED_COUNTRIES } from "@/lib/travel";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Vanchhit — Notes, projects, places" },
      {
        name: "description",
        content:
          "Personal site of Vanchhit. Writing on aviation, trains, AI safety and climate. Projects, places, and other cool stuff.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  const latest = BLOG_POSTS.slice(0, 3);

  return (
    <div>
      {/* Hero */}
      <section className="mx-auto max-w-6xl px-5 pb-16 pt-20 sm:pt-28">
        <div className="font-mono text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Vanchhit · personal site
        </div>
        <h1 className="mt-5 max-w-4xl font-display text-5xl font-semibold leading-[1.05] tracking-tight sm:text-7xl">
          A quiet corner for{" "}
          <span style={{ color: "var(--accent)" }}>projects</span>,{" "}
          <span style={{ color: "var(--warm)" }}>writing</span>,
          <br className="hidden sm:block" />
          and the places I keep going back to.
        </h1>
        <p className="mt-7 max-w-2xl text-lg text-muted-foreground sm:text-xl">
          I write about aviation, trains, AI safety, and climate — and keep a small
          map of countries I've visited and spots worth a detour.
        </p>
        <div className="mt-10 flex flex-wrap gap-3">
          <Link
            to="/blog"
            className="inline-flex items-center gap-2 rounded-full bg-foreground px-5 py-2.5 text-sm font-medium text-background transition-opacity hover:opacity-90"
          >
            Read the blog <ArrowUpRight size={14} />
          </Link>
          <Link
            to="/projects"
            className="inline-flex items-center gap-2 rounded-full border border-input px-5 py-2.5 text-sm font-medium hover:bg-muted"
          >
            See projects
          </Link>
        </div>
      </section>

      {/* Topic strip */}
      <section className="border-y border-border/60 bg-muted/40">
        <div className="mx-auto grid max-w-6xl grid-cols-2 gap-px overflow-hidden px-0 sm:grid-cols-5">
          {TOPICS.map((t) => (
            <Link
              key={t.slug}
              to={t.slug === "general" ? "/blog" : `/blog/${t.slug}` as "/blog"}
              data-topic={t.slug}
              className="group flex flex-col gap-1 bg-background px-5 py-6 transition-colors hover:bg-accent-soft"
            >
              <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                Topic
              </div>
              <div className="font-display text-base font-semibold tracking-tight" style={{ color: "var(--accent)" }}>
                {t.short}
              </div>
              <div className="text-xs text-muted-foreground">{t.description}</div>
            </Link>
          ))}
        </div>
      </section>

      {/* Latest writing */}
      <section className="mx-auto max-w-6xl px-5 py-20">
        <div className="flex items-end justify-between">
          <h2 className="font-display text-3xl font-semibold tracking-tight">Latest writing</h2>
          <Link to="/blog" className="text-sm text-muted-foreground hover:text-foreground">
            All posts →
          </Link>
        </div>
        <div className="mt-10 grid grid-cols-1 gap-6 md:grid-cols-3">
          {latest.length === 0 && (
            <div className="text-sm text-muted-foreground">No posts yet — add one in <code className="font-mono">content/blog/</code>.</div>
          )}
          {latest.map((p) => (
            <Link
              key={p.slug}
              to="/blog/$slug"
              params={{ slug: p.slug }}
              data-topic={p.topic}
              className="group flex flex-col rounded-xl border border-border/60 bg-card p-6 transition-all hover:border-accent hover:shadow-sm"
            >
              <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: "var(--accent)" }}>
                {topicLabel(p.topic)}
              </div>
              <div className="mt-3 font-display text-lg font-semibold leading-snug tracking-tight">
                {p.title}
              </div>
              <div className="mt-2 text-sm text-muted-foreground">{p.excerpt}</div>
              <div className="mt-6 font-mono text-xs text-muted-foreground">{p.date}</div>
            </Link>
          ))}
        </div>
      </section>

      {/* Travel teaser */}
      <section className="border-t border-border/60 bg-muted/30">
        <div className="mx-auto grid max-w-6xl grid-cols-1 gap-10 px-5 py-20 md:grid-cols-2 md:items-center">
          <div>
            <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Travel
            </div>
            <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight">
              {VISITED_COUNTRIES.length} countries, and counting.
            </h2>
            <p className="mt-4 max-w-md text-muted-foreground">
              An interactive world map of where I've been, plus pins for spots I'd
              send a friend back to.
            </p>
            <Link
              to="/travel"
              className="mt-6 inline-flex items-center gap-2 text-sm font-medium hover:gap-3 transition-all"
              style={{ color: "var(--accent)" }}
            >
              Open the map <ArrowUpRight size={14} />
            </Link>
          </div>
          <div className="aspect-[4/3] rounded-2xl border border-border/60 bg-background motif-blueprint" />
        </div>
      </section>
    </div>
  );
}
