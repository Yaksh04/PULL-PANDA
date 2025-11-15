import { z } from "zod";

export const repositorySchema = z.object({
  id: z.number(),
  name: z.string(),
  full_name: z.string(),
  description: z.string().nullable(),
  private: z.boolean(),
  html_url: z.string(),
  stargazers_count: z.number(),
  forks_count: z.number(),
  language: z.string().nullable(),
  open_issues_count: z.number(),
  updated_at: z.string(),
});

export const pullRequestSchema = z.object({
  id: z.number(),
  number: z.number(),
  title: z.string(),
  state: z.enum(['open', 'closed']),
  merged: z.boolean().optional(),
  html_url: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  user: z.object({
    login: z.string(),
    avatar_url: z.string(),
  }),
  repository: z.string(),
});

export const reviewCommentSchema = z.object({
  id: z.number(),
  body: z.string(),
  user: z.object({
    login: z.string(),
    avatar_url: z.string(),
  }),
  created_at: z.string(),
});

export const userSchema = z.object({
  login: z.string(),
  avatar_url: z.string(),
  name: z.string().nullable(),
  email: z.string().nullable(),
  bio: z.string().nullable(),
  public_repos: z.number(),
});

export type Repository = z.infer<typeof repositorySchema>;
export type PullRequest = z.infer<typeof pullRequestSchema>;
export type ReviewComment = z.infer<typeof reviewCommentSchema>;
export type GitHubUser = z.infer<typeof userSchema>;
