export interface Repository {
  id: number;
  name: string;
  owner: string;

  description?: string;
  private: boolean;
  html_url: string;

  stargazers_count: number;
  forks_count: number;
  language?: string;

  open_issues_count: number;
  open_prs_count: number; // ✅ added

  updated_at: string;
}

export interface PullRequest {
  id: number;
  number: number;
  title: string;
  state: 'open' | 'closed' | 'merged';
  merged: boolean;
  html_url: string;
  created_at: string;
  updated_at: string;
  user: {
    login: string;
    avatar_url: string;
  };
  repository: string;
  owner: string;
}

export interface GitHubUser {
  login: string;
  avatar_url: string;
  name: string | null;
  email: string | null;
  bio: string | null;
  public_repos: number;
}

export interface Stats {
  totalPRs: number;
  openPRs: number;
  mergedPRs: number;
  closedPRs: number;
  acceptanceRate: number;
  activeRepos: number;
}

export interface PRReviews {
  reviews: any[];
  aiReviews: any[];
  allComments: any[];
}
