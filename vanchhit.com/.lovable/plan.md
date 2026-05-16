
# vanchhit.com — Plan

A personal site with an Apple-minimal shell, persona-shifting accents per blog topic, markdown-driven content, and static hosting on GitHub Pages at your custom domain.

## 1. Design System

**Shell (global, fixed):**
- Apple-like minimal: generous whitespace, large quiet typography, restrained motion, subtle dividers, no busy gradients.
- Auto theme: respects `prefers-color-scheme`, with a manual light/dark toggle that overrides and persists in `localStorage`.
- Type pairing (Google Fonts): **Inter Tight** (display, tight tracking) + **Inter** (body) + **JetBrains Mono** (captions, metadata). Loaded via `<link rel="preconnect">` + `&display=swap`.
- Base palette: warm off-white / near-black neutrals + a primary blue accent with a secondary warm orange/yellow accent.

**Persona accents (per blog topic):**
Each topic owns CSS custom-properties scoped to a `data-topic="..."` attribute. The shell stays identical; only `--accent`, `--accent-soft`, hero treatment, and a small motif change.

| Topic | Accent | Soft bg | Motif |
|---|---|---|---|
| AI Safety | Clean indigo `#4f46e5` | `#eef2ff` | Thin grid, mono captions |
| Climate Change | Earth green `#3f6b3a` + clay `#c2956b` | `#f5f0e8` | Topographic line texture |
| Aviation / Plane Spotting | Sky blue `#3b82f6` | blueprint cyan `#e0f2fe` | Faint blueprint grid + crosshair |
| Trains / Transport | Retro rail amber `#d4842a` on charcoal | cream `#f5f0e8` | Ticket-stub dividers, serif numerals |
| Default / Other | Primary blue + orange | paper | Plain |

All accents defined as tokens in `src/styles.css` so components never hardcode color.

## 2. Information Architecture & Routes

Each section is a real route (SSR/SEO friendly), not a hash anchor.

```text
/                       Home — short hero, latest 3 posts, projects teaser, world map preview
/about                  About me — long-form, photo, timeline
/projects               Projects grid + per-project pages /projects/$slug
/interests              Interests overview (links into blog topics + games + map)
/travel                 SVG world map of countries visited + Leaflet map of favourite places
/games                  Games section (embeds / links / thumbnails)
/cool-stuff             Catch-all for "other cool stuff"; easy to add cards
/blog                   Main blog feed with filters (topic, year, search)
/blog/aviation          Topic feed (persona-themed)
/blog/trains
/blog/ai-safety
/blog/climate
/blog/$slug             Individual post (inherits topic theme from frontmatter)
/now                    Optional "what I'm doing now" page (easy to add)
```

Header nav: About · Projects · Travel · Blog · Games · More. Sticky, minimal, with theme toggle.

## 3. Tech Stack

- **Framework:** Keep the current TanStack Start + Vite + React 19 template, configured to **prerender all routes to static HTML**. Output goes to `dist/`, fully static — no server needed at runtime.
- **Styling:** Tailwind v4 via `src/styles.css` with semantic tokens (`--background`, `--foreground`, `--accent`, `--accent-soft`, persona overrides scoped by `[data-topic]`).
- **Content:** Markdown/MDX files in `/content/`, parsed at build with `vite-plugin-md` style approach using `gray-matter` + `remark`/`rehype` (or `@mdx-js/rollup` if you want JSX in posts). A small build step generates `src/content/index.json` with metadata so the filter UI is fast and client-only.
- **Maps:**
  - Countries: `react-simple-maps` with a TopoJSON world atlas; `visited` array drives fill color.
  - Places: `react-leaflet` + OpenStreetMap tiles; pins from `/content/places.json` with title, photo, short note.
- **Search/filters:** client-side over the generated JSON (Fuse.js for fuzzy search across blog).
- **Icons:** `lucide-react`.
- **Motion:** `motion/react` for restrained fades and hero reveals only.
- **Analytics (optional):** Plausible script (privacy-friendly, no cookies).

## 4. Backend

There is **no traditional backend**. All dynamic-feeling parts are derived from files in the repo at build time:

- Blog posts → `/content/blog/*.md` with frontmatter (`title, date, topic, tags, excerpt, cover`).
- Projects → `/content/projects/*.md`.
- Travel → `/content/countries.json` + `/content/places.json`.
- Cool stuff & games → `/content/cool-stuff/*.md`, `/content/games.json`.

A `scripts/build-content.ts` runs in `prebuild`, validates frontmatter with Zod, and writes `src/generated/content.ts`. This keeps types strict and the runtime free of file-system access (so it works on GitHub Pages).

If later you want comments, contact form, or newsletter, the cleanest add-on is a single Cloudflare Worker or Lovable Cloud function — site stays static otherwise.

## 5. Hosting on GitHub Pages at vanchhit.com

- Repo: `vanchhit/vanchhit.github.io` (or any repo with Pages enabled from `gh-pages` branch).
- GitHub Action `.github/workflows/deploy.yml`: on push to `main`, run `bun install && bun run build`, then deploy `dist/` to `gh-pages` via `peaceiris/actions-gh-pages`.
- Add `public/CNAME` containing `vanchhit.com`.
- DNS at your registrar:
  - `A` records for `@` → GitHub Pages IPs `185.199.108.153`, `.109.153`, `.110.153`, `.111.153`
  - `CNAME` for `www` → `vanchhit.github.io`
- Enable HTTPS in repo Settings → Pages.

## 6. Guide.md (created at repo root)

Will cover, in plain language:
1. **Hosting:** repo setup, Action, CNAME, DNS records, enabling HTTPS, propagation.
2. **Adding a blog post:** create `content/blog/my-post.md`, frontmatter template, choosing `topic` to get persona theme, adding cover image to `public/blog/`, commit → auto-deploy.
3. **Adding a project:** `content/projects/my-project.md` template + screenshots folder.
4. **Adding a country / favourite place:** edit `content/countries.json` (ISO-3 codes) and `content/places.json` (lat, lng, title, note, photo).
5. **Adding a "cool stuff" card or a game:** drop a markdown file or JSON entry — no code changes needed.
6. **Adding a whole new section/route:** create `src/routes/<name>.tsx` from a provided template, add a nav link, optional content folder.
7. **Tweaking design:** where tokens live (`src/styles.css`), how to add a new persona theme, light/dark behavior, font swap.
8. **Local development:** `bun install`, `bun run dev`, preview, build.

## 7. Build Order

1. Design tokens + light/dark + system preference + toggle + Google Fonts wired in.
2. Shell: header, footer, layout primitives, persona theming mechanism.
3. Home, About, Projects (static placeholder content + markdown loader).
4. Blog: markdown pipeline, list, filters, single post, topic feeds with persona accents.
5. Travel: SVG world map + Leaflet places map.
6. Games + Cool Stuff sections.
7. SEO: per-route `head()` metadata, sitemap.xml, robots.txt, og images.
8. GitHub Action + CNAME + Guide.md.
9. QA pass: lighthouse, mobile, dark mode, all routes prerender.

## Open items I'd like you to confirm before I build

- **Repo name** you'll use (e.g. `vanchhit.github.io` vs `personal-site`) — affects the deploy workflow's base path.
- **Initial content seed:** OK if I scaffold with 2–3 placeholder blog posts per topic, ~3 placeholder projects, and a starter `places.json` with a couple of pins so the site looks alive on first deploy? You replace later via the Guide.
- **Photo of you** for About + an OG share image — upload now, or use a tasteful placeholder for now.
- **Games section:** are these games you've built (link/embed to your projects), games you like, or playable embeds? Affects layout.
