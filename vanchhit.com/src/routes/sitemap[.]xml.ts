import { createFileRoute } from "@tanstack/react-router";
import type {} from "@tanstack/react-start";
import { BLOG_POSTS, PROJECTS } from "@/lib/content";

const BASE_URL = "https://vanchhit.com";

export const Route = createFileRoute("/sitemap.xml")({
  server: {
    handlers: {
      GET: async () => {
        const staticPaths = [
          "/", "/about", "/projects", "/travel", "/blog",
          "/blog/aviation", "/blog/trains", "/blog/ai-safety", "/blog/climate",
          "/games", "/cool-stuff",
        ];
        const blogPaths = BLOG_POSTS.map((p) => `/blog/${p.slug}`);
        const projectPaths = PROJECTS.map((p) => `/projects/${p.slug}`);

        const urls = [...staticPaths, ...blogPaths, ...projectPaths]
          .map((p) => `  <url><loc>${BASE_URL}${p}</loc></url>`)
          .join("\n");

        const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>`;

        return new Response(xml, {
          headers: { "Content-Type": "application/xml", "Cache-Control": "public, max-age=3600" },
        });
      },
    },
  },
});
