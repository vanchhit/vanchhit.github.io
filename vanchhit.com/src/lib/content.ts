import { marked } from "marked";
import type { TopicSlug } from "./topics";

// Tiny browser-safe YAML-ish frontmatter parser.
// Supports: key: value, key: "quoted value", key: ['a','b'] arrays, comments.
// Keep frontmatter simple — see Guide.md for the canonical format.
function parseFrontmatter(raw: string): { data: Record<string, unknown>; body: string } {
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) return { data: {}, body: raw };
  const [, fmRaw, body] = match;
  const data: Record<string, unknown> = {};
  for (const line of fmRaw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const colon = trimmed.indexOf(":");
    if (colon === -1) continue;
    const key = trimmed.slice(0, colon).trim();
    let val: string = trimmed.slice(colon + 1).trim();
    if (!val) {
      data[key] = "";
      continue;
    }
    // Array: [a, b, "c"]
    if (val.startsWith("[") && val.endsWith("]")) {
      const inner = val.slice(1, -1).trim();
      data[key] = inner
        ? inner.split(",").map((s) => s.trim().replace(/^["']|["']$/g, ""))
        : [];
      continue;
    }
    // Quoted string
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    // Boolean
    if (val === "true" || val === "false") {
      data[key] = val === "true";
      continue;
    }
    data[key] = val;
  }
  return { data, body };
}

export type BlogPost = {
  slug: string;
  title: string;
  date: string; // ISO YYYY-MM-DD
  topic: TopicSlug;
  tags: string[];
  excerpt: string;
  cover?: string;
  body: string; // raw markdown
  html: string; // rendered html
};

export type Project = {
  slug: string;
  title: string;
  year: string;
  role?: string;
  url?: string;
  repo?: string;
  excerpt: string;
  cover?: string;
  tags: string[];
  body: string;
  html: string;
};

export type CoolThing = {
  slug: string;
  title: string;
  url?: string;
  category?: string;
  excerpt: string;
  body: string;
  html: string;
};

marked.setOptions({ gfm: true, breaks: false });

function renderMd(body: string): string {
  return marked.parse(body, { async: false }) as string;
}

// ---------- Blog posts ----------
const blogModules = import.meta.glob("/content/blog/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

export const BLOG_POSTS: BlogPost[] = Object.entries(blogModules)
  .map(([path, raw]) => {
    const { data, body } = parseFrontmatter(raw);
    const slug = path.split("/").pop()!.replace(/\.md$/, "");
    return {
      slug,
      title: String(data.title ?? slug),
      date: String(data.date ?? "1970-01-01"),
      topic: (data.topic as TopicSlug) ?? "general",
      tags: Array.isArray(data.tags) ? (data.tags as string[]) : [],
      excerpt: String(data.excerpt ?? ""),
      cover: data.cover ? String(data.cover) : undefined,
      body,
      html: renderMd(body),
    };
  })
  .sort((a, b) => (a.date < b.date ? 1 : -1));

export function getPost(slug: string): BlogPost | undefined {
  return BLOG_POSTS.find((p) => p.slug === slug);
}

export function postsByTopic(topic: TopicSlug): BlogPost[] {
  return BLOG_POSTS.filter((p) => p.topic === topic);
}

// ---------- Projects ----------
const projectModules = import.meta.glob("/content/projects/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

export const PROJECTS: Project[] = Object.entries(projectModules)
  .map(([path, raw]) => {
    const { data, body } = parseFrontmatter(raw);
    const slug = path.split("/").pop()!.replace(/\.md$/, "");
    return {
      slug,
      title: String(data.title ?? slug),
      year: String(data.year ?? ""),
      role: data.role ? String(data.role) : undefined,
      url: data.url ? String(data.url) : undefined,
      repo: data.repo ? String(data.repo) : undefined,
      excerpt: String(data.excerpt ?? ""),
      cover: data.cover ? String(data.cover) : undefined,
      tags: Array.isArray(data.tags) ? (data.tags as string[]) : [],
      body,
      html: renderMd(body),
    };
  })
  .sort((a, b) => (a.year < b.year ? 1 : -1));

export function getProject(slug: string): Project | undefined {
  return PROJECTS.find((p) => p.slug === slug);
}

// ---------- Cool stuff ----------
const coolModules = import.meta.glob("/content/cool-stuff/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

export const COOL_THINGS: CoolThing[] = Object.entries(coolModules)
  .map(([path, raw]) => {
    const { data, body } = parseFrontmatter(raw);
    const slug = path.split("/").pop()!.replace(/\.md$/, "");
    return {
      slug,
      title: String(data.title ?? slug),
      url: data.url ? String(data.url) : undefined,
      category: data.category ? String(data.category) : undefined,
      excerpt: String(data.excerpt ?? ""),
      body,
      html: renderMd(body),
    };
  })
  .sort((a, b) => (a.title < b.title ? -1 : 1));
