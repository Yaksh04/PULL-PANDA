// SWE_project_website/client/src/pages/Analytics.tsx
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { StatsCard } from "@/components/StatsCard";
import { ErrorState } from "@/components/ErrorState";
import { Skeleton } from "@/components/ui/skeleton";
import { TrendingUp, Target, Clock, Activity } from "lucide-react";
import { queryClient } from "@/lib/queryClient";
import type { Stats, PullRequest } from "@/lib/api";

export default function Analytics() {

  // ✅ FIXED — Stats query now performs real fetch
  const { data: stats, isLoading: statsLoading, error: statsError } = useQuery<Stats>({
    queryKey: ['/api/stats'],
    queryFn: async () => {
      const res = await fetch("http://localhost:5000/api/stats", {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to load stats");
      return res.json();
    }
  });

  // ✅ FIXED — Pull requests query now works
  const { data: prs, error: prsError } = useQuery<PullRequest[]>({
    queryKey: ['/api/pull-requests'],
    queryFn: async () => {
      const res = await fetch("http://localhost:5000/api/pull-requests", {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to load pull requests");
      return res.json();
    }
  });

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['/api/stats'] });
    queryClient.invalidateQueries({ queryKey: ['/api/pull-requests'] });
  };

  const repoStats = prs?.reduce((acc, pr) => {
    acc[pr.repository] = (acc[pr.repository] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const topRepos = Object.entries(repoStats || {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, 4);

  const mergedPRs = stats?.mergedPRs || 0;
  const totalPRs = stats?.totalPRs || 1;
  const openPRs = stats?.openPRs || 0;
  const closedPRs = stats?.closedPRs || 0;

  const approvedPercent = Math.round((mergedPRs / totalPRs) * 100);
  const changesPercent = Math.round((closedPRs / totalPRs) * 100);
  const commentedPercent = Math.round((openPRs / totalPRs) * 100);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto p-6 space-y-6">
          <div>
            <h1 className="text-2xl font-semibold text-foreground">Analytics</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Review performance metrics and trends
            </p>
          </div>

          {statsError ? (
            <ErrorState
              title="Failed to load analytics"
              message="Unable to fetch analytics data from GitHub. Please check your connection and try again."
              onRetry={handleRefresh}
            />
          ) : (
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
                    title="Review Quality Score"
                    value={`${approvedPercent}%`}
                    icon={Target}
                    trend="Based on merged PRs"
                  />
                  <StatsCard
                    title="Total Reviews"
                    value={stats.totalPRs}
                    icon={Activity}
                    trend={`${stats.openPRs} in progress`}
                  />
                  <StatsCard
                    title="Merged PRs"
                    value={stats.mergedPRs}
                    icon={TrendingUp}
                    trend={`${stats.acceptanceRate}% rate`}
                  />
                  <StatsCard
                    title="Active Repos"
                    value={stats.activeRepos}
                    icon={Clock}
                    trend="Contributing to"
                  />
                </>
              ) : null}
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-card-foreground mb-4">
                PR Status Distribution
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Merged</span>
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-48 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-chart-2" style={{ width: `${approvedPercent}%` }} />
                    </div>
                    <span className="text-sm font-medium text-card-foreground w-12 text-right">{approvedPercent}%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Closed</span>
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-48 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-chart-4" style={{ width: `${changesPercent}%` }} />
                    </div>
                    <span className="text-sm font-medium text-card-foreground w-12 text-right">{changesPercent}%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Open</span>
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-48 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-chart-1" style={{ width: `${commentedPercent}%` }} />
                    </div>
                    <span className="text-sm font-medium text-card-foreground w-12 text-right">{commentedPercent}%</span>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-card-foreground mb-4">
                Top Repositories by PR Volume
              </h3>
              {topRepos.length > 0 ? (
                <div className="space-y-4">
                  {topRepos.map(([repo, count], index) => {
                    const maxCount = topRepos[0][1];
                    const percentage = (count / maxCount) * 100;
                    return (
                      <div key={repo} className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground truncate max-w-[150px]">{repo}</span>
                        <div className="flex items-center gap-2">
                          <div className="h-2 w-48 bg-muted rounded-full overflow-hidden">
                            <div 
                              className={`h-full bg-chart-${(index % 5) + 1}`}
                              style={{ width: `${percentage}%` }} 
                            />
                          </div>
                          <span className="text-sm font-medium text-card-foreground w-12 text-right">{count}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">No repository data available</p>
              )}
            </Card>
          </div>

          <Card className="p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-card-foreground">
                Future Enhancements
              </h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-muted rounded-md">
                <h4 className="text-sm font-medium text-card-foreground mb-2">
                  AI Review Quality Metrics
                </h4>
                <p className="text-xs text-muted-foreground">
                  Track accuracy, helpfulness scores, and developer satisfaction ratings
                </p>
              </div>
              <div className="p-4 bg-muted rounded-md">
                <h4 className="text-sm font-medium text-card-foreground mb-2">
                  Static Analysis Reports
                </h4>
                <p className="text-xs text-muted-foreground">
                  Detailed code quality insights with security vulnerability detection
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
