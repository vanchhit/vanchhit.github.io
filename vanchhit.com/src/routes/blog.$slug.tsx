import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { getPost } from "@/lib/content";
import { topicLabel, topicMotifClass } from "@/lib/topics";
import { ArrowLeft } from "lucide-react";

export const Route = createFileRoute("/blog/$slug")({
  loader: ({ params }) => {
    const post = getPost(params.slug);
    if (!post) throw notFound();
    return { post };
  },
  head: ({ loaderData }) => ({
    meta: loaderData
      ? [
          { title: `${loaderData.post.title} — Vanchhit` },
          { name: "description", content: loaderData.post.excerpt },
          { property: "og:title", content: loaderData.post.title },
          { property: "og:description", content: loaderData.post.excerpt },
          { property: "article:published_time", content: loaderData.post.date },
          ...(loaderData.post.cover ? [{ property: "og:image" as const, content: loaderData.post.cover }] : []),
        ]
      : [],
  }),
  component: PostPage,
  notFoundComponent: () => (
    <div className="mx-auto max-w-3xl px-5 py-20 text-center">
      <h1 className="font-display text-3xl font-semibold">Post not found</h1>
      <Link to="/blog" className="mt-4 inline-block text-sm" style={{ color: "var(--accent)" }}>← All posts</Link>
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

function PostPage() {
  const { post } = Route.useLoaderData();

  return (
    <div data-topic={post.topic}>
      <header className={`border-b border-border/60 ${topicMotifClass(post.topic)}`}>
        <div className="mx-auto max-w-3xl px-5 py-16">
          <Link to="/blog" className="inline-flex items-center gap-1.5 font-mono text-xs uppercase tracking-widest text-muted-foreground hover:text-foreground">
            <ArrowLeft size={12} /> Blog
          </Link>
          <div className="mt-6 font-mono text-xs uppercase tracking-widest" style={{ color: "var(--accent)" }}>
            {topicLabel(post.topic)} · {post.date}
          </div>
          <h1 className="mt-3 font-display text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
            {post.title}
          </h1>
          {post.excerpt && (
            <p className="mt-4 text-lg text-muted-foreground">{post.excerpt}</p>
          )}
        </div>
      </header>

      <article className="mx-auto max-w-3xl px-5 py-16">
        {post.cover && (
          <img src={post.cover} alt={post.title} className="mb-10 w-full rounded-2xl border border-border" />
        )}
        <div className="prose" dangerouslySetInnerHTML={{ __html: post.html }} />

        {post.tags.length > 0 && (
          <div className="mt-12 flex flex-wrap gap-1.5">
            {post.tags.map((t: string) => (
              <span key={t} className="rounded-full bg-muted px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                #{t}
              </span>
            ))}
          </div>
        )}
      </article>
    </div>
  );
}
