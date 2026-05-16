import { createFileRoute } from "@tanstack/react-router";
import { GAMES } from "@/lib/games";
import { ExternalLink } from "lucide-react";

export const Route = createFileRoute("/games")({
  head: () => ({
    meta: [
      { title: "Games — Vanchhit" },
      { name: "description", content: "Small games and quizzes — built, sketched, or planned." },
      { property: "og:title", content: "Games — Vanchhit" },
      { property: "og:description", content: "Small games and quizzes." },
    ],
  }),
  component: GamesPage,
});

function GamesPage() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-20">
      <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">Games</div>
      <h1 className="mt-3 font-display text-5xl font-semibold leading-tight tracking-tight">
        Tiny games.
      </h1>
      <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
        Quiet little projects, mostly for fun. A few are playable, others are still
        ideas in a notebook.
      </p>

      <div className="mt-14 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {GAMES.map((g) => {
          const Card = g.url ? "a" : "div";
          const cardProps = g.url ? { href: g.url, target: "_blank", rel: "noreferrer" } : {};
          return (
            <Card
              key={g.id}
              {...cardProps}
              className="group flex flex-col rounded-2xl border border-border/60 bg-card p-6 transition-all hover:border-accent"
            >
              <div className="text-3xl">{g.emoji}</div>
              <div className="mt-4 font-display text-xl font-semibold tracking-tight">{g.title}</div>
              <p className="mt-2 flex-1 text-sm text-muted-foreground">{g.description}</p>
              <div className="mt-5 flex items-center justify-between">
                <span className={`rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest ${
                  g.status === "playable" ? "bg-accent-soft text-accent" :
                  g.status === "in-progress" ? "bg-muted text-foreground" :
                  "bg-muted text-muted-foreground"
                }`} style={g.status === "playable" ? { color: "var(--accent)" } : undefined}>
                  {g.status}
                </span>
                {g.url && <ExternalLink size={14} className="text-muted-foreground" />}
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
