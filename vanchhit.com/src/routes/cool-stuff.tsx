import { createFileRoute } from "@tanstack/react-router";
import { COOL_THINGS } from "@/lib/content";
import { ArrowUpRight } from "lucide-react";

export const Route = createFileRoute("/cool-stuff")({
  head: () => ({
    meta: [
      { title: "Cool stuff — Vanchhit" },
      { name: "description", content: "A loose collection of links, tools, and other things worth a look." },
      { property: "og:title", content: "Cool stuff — Vanchhit" },
      { property: "og:description", content: "A loose collection of things worth a look." },
    ],
  }),
  component: CoolStuffPage,
});

function CoolStuffPage() {
  const grouped = COOL_THINGS.reduce<Record<string, typeof COOL_THINGS>>((acc, item) => {
    const k = item.category ?? "Other";
    (acc[k] ??= []).push(item);
    return acc;
  }, {});

  return (
    <div className="mx-auto max-w-6xl px-5 py-20">
      <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">More</div>
      <h1 className="mt-3 font-display text-5xl font-semibold leading-tight tracking-tight">
        Cool stuff.
      </h1>
      <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
        Links, tools, books, and other things I keep handy. A loose collection
        that grows over time.
      </p>

      {Object.keys(grouped).length === 0 && (
        <div className="mt-12 text-sm text-muted-foreground">
          Nothing here yet — add a markdown file to <code className="font-mono">content/cool-stuff/</code>.
        </div>
      )}

      <div className="mt-14 space-y-14">
        {Object.entries(grouped).map(([category, items]) => (
          <section key={category}>
            <h2 className="font-display text-xl font-semibold tracking-tight" style={{ color: "var(--accent)" }}>
              {category}
            </h2>
            <ul className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
              {items.map((item) => {
                const Card = item.url ? "a" : "div";
                const props = item.url ? { href: item.url, target: "_blank", rel: "noreferrer" } : {};
                return (
                  <Card
                    key={item.slug}
                    {...props}
                    className="group flex items-start justify-between gap-3 rounded-xl border border-border/60 bg-card p-5 transition-all hover:border-accent"
                  >
                    <div>
                      <div className="font-display text-base font-semibold tracking-tight">{item.title}</div>
                      {item.excerpt && <div className="mt-1 text-sm text-muted-foreground">{item.excerpt}</div>}
                    </div>
                    {item.url && <ArrowUpRight size={16} className="mt-0.5 shrink-0 text-muted-foreground" />}
                  </Card>
                );
              })}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
