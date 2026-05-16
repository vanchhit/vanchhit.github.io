import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — Vanchhit" },
      {
        name: "description",
        content: "About Vanchhit — what I work on, what I read, and what I'm curious about.",
      },
      { property: "og:title", content: "About — Vanchhit" },
      { property: "og:description", content: "What I work on, read, and care about." },
    ],
  }),
  component: AboutPage,
});

function AboutPage() {
  return (
    <article className="mx-auto max-w-3xl px-5 py-20">
      <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
        About
      </div>
      <h1 className="mt-4 font-display text-5xl font-semibold leading-tight tracking-tight">
        I'm Vanchhit.
      </h1>
      <p className="mt-6 text-xl text-muted-foreground">
        Curious about how things work — aircraft, trains, climate systems, and the
        models we're building. This site is where I keep notes on all of them.
      </p>

      <div className="prose mt-12">
        <p>
          Replace this with your own story. A short intro that says what you do,
          what you're working on, and why someone should care enough to keep
          reading. Two or three paragraphs is plenty.
        </p>

        <h2>Now</h2>
        <p>
          What you're spending time on this season — a short, datestamped paragraph
          that's easy to update.
        </p>

        <h2>Background</h2>
        <p>
          Where you've worked, what you've studied, and the threads that connect
          the four topics above. Keep it human.
        </p>

        <h2>Get in touch</h2>
        <p>
          Best ways to reach you — email, a couple of social links, or a contact
          form if you add one later.
        </p>
      </div>

      <div className="mt-16 rounded-xl border border-dashed border-border p-6 text-sm text-muted-foreground">
        <div className="font-mono text-[10px] uppercase tracking-widest">Editing this page</div>
        <p className="mt-2">
          This page lives at <code className="font-mono">src/routes/about.tsx</code>.
          Replace the prose above with your own. See <code className="font-mono">Guide.md</code> in the repo root.
        </p>
      </div>
    </article>
  );
}
