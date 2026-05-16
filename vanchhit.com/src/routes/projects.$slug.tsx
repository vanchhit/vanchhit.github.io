import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { getProject } from "@/lib/content";
import { ArrowLeft, ExternalLink, Github } from "lucide-react";

export const Route = createFileRoute("/projects/$slug")({
  loader: ({ params }) => {
    const project = getProject(params.slug);
    if (!project) throw notFound();
    return { project };
  },
  head: ({ loaderData }) => ({
    meta: loaderData
      ? [
          { title: `${loaderData.project.title} — Vanchhit` },
          { name: "description", content: loaderData.project.excerpt },
          { property: "og:title", content: loaderData.project.title },
          { property: "og:description", content: loaderData.project.excerpt },
          ...(loaderData.project.cover ? [{ property: "og:image" as const, content: loaderData.project.cover }] : []),
        ]
      : [],
  }),
  component: ProjectPage,
  notFoundComponent: () => (
    <div className="mx-auto max-w-3xl px-5 py-20 text-center">
      <h1 className="font-display text-3xl font-semibold">Project not found</h1>
      <Link to="/projects" className="mt-4 inline-block text-sm text-accent">← All projects</Link>
    </div>
  ),
  errorComponent: ({ error, reset }) => (
    <div className="mx-auto max-w-3xl px-5 py-20 text-center">
      <h1 className="font-display text-2xl">Something went wrong</h1>
      <p className="mt-2 text-sm text-muted-foreground">{error.message}</p>
      <button onClick={reset} className="mt-4 rounded-md border px-3 py-1.5 text-sm">Retry</button>
    </div>
  ),
});

function ProjectPage() {
  const { project } = Route.useLoaderData();
  return (
    <article className="mx-auto max-w-3xl px-5 py-20">
      <Link to="/projects" className="inline-flex items-center gap-1.5 font-mono text-xs uppercase tracking-widest text-muted-foreground hover:text-foreground">
        <ArrowLeft size={12} /> Projects
      </Link>
      <div className="mt-6 font-mono text-xs text-muted-foreground">{project.year}{project.role ? ` · ${project.role}` : ""}</div>
      <h1 className="mt-2 font-display text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
        {project.title}
      </h1>
      <p className="mt-4 text-lg text-muted-foreground">{project.excerpt}</p>

      <div className="mt-6 flex flex-wrap gap-3">
        {project.url && (
          <a href={project.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 rounded-full border border-input px-4 py-1.5 text-sm hover:bg-muted">
            <ExternalLink size={14} /> Live
          </a>
        )}
        {project.repo && (
          <a href={project.repo} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 rounded-full border border-input px-4 py-1.5 text-sm hover:bg-muted">
            <Github size={14} /> Code
          </a>
        )}
      </div>

      {project.cover && (
        <img src={project.cover} alt={project.title} className="mt-10 w-full rounded-2xl border border-border" />
      )}

      <div className="prose mt-10" dangerouslySetInnerHTML={{ __html: project.html }} />
    </article>
  );
}
