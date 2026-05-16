import { Link } from "@tanstack/react-router";

export function Footer() {
  return (
    <footer className="mt-32 border-t border-border/60">
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-10 px-5 py-12 sm:grid-cols-3">
        <div>
          <div className="font-display text-base font-semibold tracking-tight">
            vanchhit<span style={{ color: "var(--warm)" }}>.</span>
          </div>
          <p className="mt-2 max-w-xs text-sm text-muted-foreground">
            Personal site, notebook, and map. Built quietly, updated often.
          </p>
        </div>

        <div>
          <div className="text-xs uppercase tracking-widest text-muted-foreground">Read</div>
          <ul className="mt-3 space-y-1.5 text-sm">
            <li><Link to="/blog" className="hover:text-foreground">All posts</Link></li>
            <li><Link to="/blog/aviation" className="hover:text-foreground">Aviation</Link></li>
            <li><Link to="/blog/trains" className="hover:text-foreground">Trains</Link></li>
            <li><Link to="/blog/ai-safety" className="hover:text-foreground">AI Safety</Link></li>
            <li><Link to="/blog/climate" className="hover:text-foreground">Climate</Link></li>
          </ul>
        </div>

        <div>
          <div className="text-xs uppercase tracking-widest text-muted-foreground">Elsewhere</div>
          <ul className="mt-3 space-y-1.5 text-sm">
            <li><Link to="/about" className="hover:text-foreground">About</Link></li>
            <li><Link to="/projects" className="hover:text-foreground">Projects</Link></li>
            <li><Link to="/travel" className="hover:text-foreground">Travel</Link></li>
            <li><Link to="/games" className="hover:text-foreground">Games</Link></li>
          </ul>
        </div>
      </div>
      <div className="border-t border-border/60">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-2 px-5 py-5 text-xs text-muted-foreground sm:flex-row sm:items-center">
          <div>© {new Date().getFullYear()} Vanchhit. All thoughts mine.</div>
          <div className="font-mono">vanchhit.com</div>
        </div>
      </div>
    </footer>
  );
}
