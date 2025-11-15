import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { StatsCard } from "@/components/StatsCard";
import { RepositoryCard } from "@/components/RepositoryCard";
import { SearchBar } from "@/components/SearchBar";
import { ErrorState } from "@/components/ErrorState";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { GitPullRequest, CheckCircle2, Clock, FolderGit2, RefreshCw } from "lucide-react";
import { useLocation } from "wouter";
import { queryClient } from "@/lib/queryClient";
import type { Repository, Stats } from "@/lib/api";

export default function Dashboard() {
  const [, setLocation] = useLocation();
  const [searchQuery, setSearchQuery] = useState("");

  // const { data: repos, isLoading: reposLoading, error: reposError, isRefetching: reposRefetching } = useQuery<Repository[]>({
  //   queryKey: ['/api/repositories'],
  // });

  // const { data: stats, isLoading: statsLoading, error: statsError, isRefetching: statsRefetching } = useQuery<Stats>({
  //   queryKey: ['/api/stats'],
  // });

  // Fetch repositories
  const { data: repos, isLoading: reposLoading, error: reposError, isRefetching: reposRefetching } =
    useQuery<Repository[]>({
      queryKey: ['/api/repositories'],
      queryFn: async () => {
        const res = await fetch("http://localhost:5000/api/repositories", {
          credentials: "include",
        });
        if (!res.ok) throw new Error("Failed to fetch repositories");
        return res.json();
      },
    });

  // Fetch stats
  const { data: stats, isLoading: statsLoading, error: statsError, isRefetching: statsRefetching } =
    useQuery<Stats>({
      queryKey: ['/api/stats'],
      queryFn: async () => {
        const res = await fetch("http://localhost:5000/api/stats", {
          credentials: "include",
        });
        if (!res.ok) throw new Error("Failed to fetch stats");
        return res.json();
      },
    });

  const filteredRepos = repos?.filter(repo =>
    repo.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    repo.description?.toLowerCase().includes(searchQuery.toLowerCase())
  ) || [];

  const isRefreshing = reposRefetching || statsRefetching;

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['/api/repositories'] });
    queryClient.invalidateQueries({ queryKey: ['/api/stats'] });
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto p-6 space-y-6">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-2xl font-semibold text-foreground">My Repositories</h1>
              <p className="text-sm text-muted-foreground mt-1">
                Overview of your repositories and AI review activity
              </p>
            </div>
            <Button
              variant="outline"
              onClick={handleRefresh}
              disabled={isRefreshing}
              data-testid="button-refresh"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {statsLoading ? (
              <>
                <Skeleton className="h-32" />
                <Skeleton className="h-32" />
                <Skeleton className="h-32" />
                <Skeleton className="h-32" />
              </>
            ) : stats ? (
              <>
                <StatsCard
                  title="Total PRs Reviewed"
                  value={stats.totalPRs}
                  icon={GitPullRequest}
                  trend={`${stats.openPRs} open`}
                />
                <StatsCard
                  title="Acceptance Rate"
                  value={`${stats.acceptanceRate}%`}
                  icon={CheckCircle2}
                  trend={`${stats.mergedPRs} merged`}
                />
                <StatsCard
                  title="Active Repos"
                  value={stats.activeRepos}
                  icon={FolderGit2}
                />
                <StatsCard
                  title="Open PRs"
                  value={stats.openPRs}
                  icon={Clock}
                  trend={`${stats.closedPRs} closed`}
                />
              </>
            ) : null}
          </div>

          <div className="flex items-center gap-4">
            <SearchBar
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search repositories..."
            />
          </div>

          {reposError ? (
            <ErrorState
              title="Failed to load repositories"
              message="Unable to fetch your GitHub repositories. Please check your connection and try again."
              onRetry={handleRefresh}
            />
          ) : reposLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-40" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredRepos.map((repo) => (
                <RepositoryCard
                  key={repo.id}
                  name={repo.name}
                  description={repo.description || "No description"}
                  stars={repo.stargazers_count}
                  forks={repo.forks_count}
                  openPRs={repo.open_prs_count}
                  language={repo.language || undefined}
                  isPrivate={repo.private}
                  onClick={() => {
                    setLocation(`/pull-requests?repo=${repo.name}&owner=${repo.owner}`);
                  }}
                />
              ))}
            </div>
          )}

          {!reposLoading && !reposError && filteredRepos.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                {searchQuery ? "No repositories found" : "No repositories available"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
