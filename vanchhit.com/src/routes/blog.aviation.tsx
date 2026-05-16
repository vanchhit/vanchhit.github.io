import { createFileRoute, Link } from "@tanstack/react-router";
import { postsByTopic } from "@/lib/content";
import { TOPIC_BY_SLUG, topicMotifClass, type TopicSlug } from "@/lib/topics";
import { ArrowLeft } from "lucide-react";

// To add a new topic feed: copy this file to src/routes/blog.<slug>.tsx,
// change the route path, and the TOPIC constant.
const TOPIC: TopicSlug = "aviation";

export const Route = createFileRoute("/blog/aviation")({
  head: () => {
    const t = TOPIC_BY_SLUG[TOPIC];
    return {
      meta: [
        { title: `${t.label} — Vanchhit` },
        { name: "description", content: t.description },
        { property: "og:title", content: `${t.label} — Vanchhit` },
        { property: "og:description", content: t.description },
      ],
    };
  },
  component: TopicFeed,
});

function TopicFeed() {
  const topic = TOPIC_BY_SLUG[TOPIC];
  const posts = postsByTopic(TOPIC);
  return (
    <div data-topic={TOPIC}>
      <header className={`border-b border-border/60 ${topicMotifClass(TOPIC)}`}>
        <div className="mx-auto max-w-6xl px-5 py-20">
          <Link to="/blog" className="inline-flex items-center gap-1.5 font-mono text-xs uppercase tracking-widest text-muted-foreground hover:text-foreground">
            <ArrowLeft size={12} /> All posts
          </Link>
          <div className="mt-6 font-mono text-xs uppercase tracking-widest" style={{ color: "var(--accent)" }}>
            Topic
          </div>
          <h1 className="mt-2 font-display text-5xl font-semibold leading-tight tracking-tight">{topic.label}</h1>
          <p className="mt-4 max-w-2xl text-lg text-muted-foreground">{topic.description}</p>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-5 py-16">
        {posts.length === 0 ? (
          <div className="text-sm text-muted-foreground">No posts in this topic yet.</div>
        ) : (
          <div className="divide-y divide-border/60">
            {posts.map((p) => (
              <Link
                key={p.slug}
                to="/blog/$slug"
                params={{ slug: p.slug }}
                data-topic={p.topic}
                className="group grid grid-cols-1 gap-2 py-6 sm:grid-cols-[120px_1fr] sm:gap-6"
              >
                <div className="font-mono text-xs text-muted-foreground">{p.date}</div>
                <div>
                  <div className="font-display text-xl font-semibold tracking-tight transition-colors group-hover:text-accent">
                    {p.title}
                  </div>
                  <div className="mt-1 text-sm text-muted-foreground">{p.excerpt}</div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
