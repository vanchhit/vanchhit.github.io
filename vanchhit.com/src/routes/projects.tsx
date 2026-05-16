import { createFileRoute, Link } from "@tanstack/react-router";
import { PROJECTS } from "@/lib/content";
import { ArrowUpRight } from "lucide-react";

export const Route = createFileRoute("/projects")({
  head: () => ({
    meta: [
      { title: "Projects — Vanchhit" },
      { name: "description", content: "Things I've built — selected projects with notes and links." },
      { property: "og:title", content: "Projects — Vanchhit" },
      { property: "og:description", content: "Things I've built." },
    ],
  }),
  component: ProjectsPage,
});

function ProjectsPage() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-20">
      <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">Projects</div>
      <h1 className="mt-3 font-display text-5xl font-semibold leading-tight tracking-tight">
        Things I've built.
      </h1>
      <p className="mt-5 max-w-2xl text-lg text-muted-foreground">
        A selection — research, side projects, experiments. More notes in the blog.
      </p>

      <div className="mt-14 grid grid-cols-1 gap-6 md:grid-cols-2">
        {PROJECTS.length === 0 && (
          <div className="text-sm text-muted-foreground">
            No projects yet. Add one to <code className="font-mono">content/projects/</code>.
          </div>
        )}
        {PROJECTS.map((p) => (
          <Link
            key={p.slug}
            to="/projects/$slug"
            params={{ slug: p.slug }}
            className="group flex flex-col rounded-2xl border border-border/60 bg-card p-7 transition-all hover:border-accent hover:shadow-sm"
          >
            <div className="flex items-center justify-between">
              <div className="font-mono text-xs text-muted-foreground">{p.year}</div>
              <ArrowUpRight size={16} className="text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
            </div>
            <h2 className="mt-3 font-display text-2xl font-semibold tracking-tight">{p.title}</h2>
            {p.role && <div className="mt-1 text-sm text-muted-foreground">{p.role}</div>}
            <p className="mt-3 text-sm text-muted-foreground">{p.excerpt}</p>
            {p.tags.length > 0 && (
              <div className="mt-5 flex flex-wrap gap-1.5">
                {p.tags.map((t) => (
                  <span key={t} className="rounded-full bg-muted px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                    {t}
                  </span>
                ))}
              </div>
            )}
          </Link>
        ))}
      </div>
    </div>
  );
}
