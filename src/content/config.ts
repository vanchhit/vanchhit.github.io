import { defineCollection, z } from 'astro:content';

export const BLOG_CATEGORIES = [
  'aviation',
  'trains',
  'ai-safety',
  'climate',
  'tech',
  'personal',
] as const;

export type BlogCategory = (typeof BLOG_CATEGORIES)[number];

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date(),
    categories: z.array(z.enum(BLOG_CATEGORIES)),
    tags: z.array(z.string()).default([]),
    description: z.string(),
    image: z.string().optional(),
    featured: z.boolean().default(false),
    draft: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    tags: z.array(z.string()).default([]),
    github: z.string().url().optional(),
    live: z.string().url().optional(),
    featured: z.boolean().default(false),
    date: z.string(),
    status: z.enum(['active', 'complete', 'archived']).default('complete'),
  }),
});

const rulebooks = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
  }),
});

export const collections = { blog, projects, rulebooks };
