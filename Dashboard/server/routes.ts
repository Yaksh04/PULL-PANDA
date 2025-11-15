// SWE_project_website/server/routes.ts
import type { Express, Request, Response } from "express";
import { Octokit } from "@octokit/rest";

// Typed helper to get authenticated GitHub client
function getClient(req: Request): Octokit {
  const token = (req.session as any)?.accessToken;

  if (!token) {
    const error: any = new Error("Not authenticated");
    error.status = 401;
    throw error;
  }

  return new Octokit({ auth: token });
}

export async function registerRoutes(app: Express): Promise<void> {
  // --------------------------
  // Current Logged-in User
  // --------------------------
  app.get("/api/user", async (req: Request, res: Response) => {
    try {
      const octokit = getClient(req);
      const { data: user } = await octokit.rest.users.getAuthenticated();
      res.json(user);
    } catch (error: any) {
      console.error("Error fetching user:", error);
      res.status(error.status || 500).json({ error: error.message });
    }
  });

  // --------------------------
  // Repositories
  // --------------------------
  app.get("/api/repositories", async (req: Request, res: Response) => {
    try {
      const octokit = getClient(req);

      const { data: repos } = await octokit.rest.repos.listForAuthenticatedUser({
        sort: "updated",
        per_page: 30,
      });

      // To compute open PR count per repo:
      const repositoriesWithPRCounts = await Promise.all(
        repos.map(async (repo) => {
          try {
            const { data: pullRequests } = await octokit.rest.pulls.list({
              owner: repo.owner!.login,
              repo: repo.name,
              state: "open",
            });

            return {
              id: repo.id,
              name: repo.name,
              owner: repo.owner?.login || "",
              full_name: repo.full_name,
              description: repo.description,
              private: repo.private,
              html_url: repo.html_url,

              stargazers_count: repo.stargazers_count,
              forks_count: repo.forks_count,
              language: repo.language,

              open_issues_count: repo.open_issues_count,
              open_prs_count: pullRequests.length, // ✅ real number of open PRs

              updated_at: repo.updated_at,
            };
          } catch (err) {
            console.error(`Failed to fetch PR count for ${repo.name}`, err);

            return {
              id: repo.id,
              name: repo.name,
              owner: repo.owner?.login || "",
              full_name: repo.full_name,
              description: repo.description,
              private: repo.private,
              html_url: repo.html_url,

              stargazers_count: repo.stargazers_count,
              forks_count: repo.forks_count,
              language: repo.language,

              open_issues_count: repo.open_issues_count,
              open_prs_count: 0, // fallback

              updated_at: repo.updated_at,
            };
          }
        })
      );

      res.json(repositoriesWithPRCounts);

    } catch (error: any) {
      console.error("Error fetching repositories:", error);
      res.status(error.status || 500).json({ error: error.message });
    }
  });


  // --------------------------
  // Pull Requests
  // --------------------------
  app.get("/api/pull-requests", async (req: Request, res: Response) => {
    try {
      const octokit = getClient(req);
      const { data: repos } = await octokit.rest.repos.listForAuthenticatedUser({
        per_page: 100
      });

      const allPRs: any[] = [];

      for (const repo of repos.slice(0, 20)) {
        try {
          const { data: prs } = await octokit.rest.pulls.list({
            owner: repo.owner!.login,
            repo: repo.name,
            state: "all",
            per_page: 20,
            sort: "updated",
            direction: "desc"
          });

          prs.forEach((pr) => {
            allPRs.push({
              id: pr.id,
              number: pr.number,
              title: pr.title,
              state: pr.state,
              merged: pr.merged_at !== null,
              html_url: pr.html_url,
              created_at: pr.created_at,
              updated_at: pr.updated_at,
              repository: repo.name,
              owner: repo.owner!.login,

              // ✅ ADD THESE:
              user: {
                login: pr.user?.login || "unknown",
                avatar_url: pr.user?.avatar_url || "",
              }
            });
          });

        } catch (innerErr) {
          console.error(`PR fetch error for repo ${repo.name}:`, innerErr);
        }
      }

      allPRs.sort(
        (a, b) =>
          new Date(b.updated_at).getTime() -
          new Date(a.updated_at).getTime()
      );

      res.json(allPRs);
    } catch (error: any) {
      res.status(error.status || 500).json({ error: error.message });
    }
  });

  // --------------------------
  // PR Reviews
  // --------------------------
  app.get(
    "/api/pull-requests/:owner/:repo/:number/reviews",
    async (req: Request, res: Response) => {
      try {
        const octokit = getClient(req);
        const { owner, repo, number } = req.params;
        const PR_NUMBER = parseInt(number);

        const { data: reviews } = await octokit.rest.pulls.listReviews({
          owner,
          repo,
          pull_number: PR_NUMBER
        });

        const { data: comments } = await octokit.rest.issues.listComments({
          owner,
          repo,
          issue_number: PR_NUMBER
        });

        const aiReviews = comments.filter(
          (c) =>
            c.user?.login?.toLowerCase().includes("bot") ||
            c.body?.toLowerCase().includes("ai review") ||
            c.body?.toLowerCase().includes("static analysis")
        );

        res.json({
          reviews,
          aiReviews,
          allComments: comments
        });
      } catch (error: any) {
        res.status(error.status || 500).json({ error: error.message });
      }
    }
  );

  // --------------------------
  // Stats
  // --------------------------
  app.get("/api/stats", async (req: Request, res: Response) => {
    try {
      const octokit = getClient(req);

      const { data: repos } = await octokit.rest.repos.listForAuthenticatedUser({
        per_page: 100
      });

      let totalPRs = 0;
      let openPRs = 0;
      let mergedPRs = 0;
      let closedPRs = 0;

      for (const repo of repos.slice(0, 20)) {
        try {
          const { data: prs } = await octokit.rest.pulls.list({
            owner: repo.owner!.login,
            repo: repo.name,
            state: "all",
            per_page: 100
          });

          totalPRs += prs.length;
          openPRs += prs.filter((pr) => pr.state === "open").length;
          mergedPRs += prs.filter((pr) => pr.merged_at).length;
          closedPRs += prs.filter((pr) => pr.state === "closed" && !pr.merged_at).length;
        } catch (statsErr) {
          console.error(`Stats error for ${repo.name}:`, statsErr);
        }
      }

      res.json({
        totalPRs,
        openPRs,
        mergedPRs,
        closedPRs,
        acceptanceRate: totalPRs
          ? Math.round((mergedPRs / totalPRs) * 100)
          : 0,
        activeRepos: repos.length
      });
    } catch (error: any) {
      res.status(error.status || 500).json({ error: error.message });
    }
  });
}
