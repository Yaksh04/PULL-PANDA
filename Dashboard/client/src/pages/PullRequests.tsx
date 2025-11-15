// SWE_project_website/client/src/pages/PullRequests.tsx
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { PullRequestCard } from "@/components/PullRequestCard";
import { SearchBar } from "@/components/SearchBar";
import { FilterBar } from "@/components/FilterBar";
import { ErrorState } from "@/components/ErrorState";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw } from "lucide-react";
import { queryClient } from "@/lib/queryClient";
import type { PullRequest } from "@/lib/api";

export default function PullRequests() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filters, setFilters] = useState([
    { id: "open", label: "Open", active: true },
    { id: "merged", label: "Merged", active: false },
    { id: "closed", label: "Closed", active: false },
  ]);

  // ✅ FIXED: Added queryFn with credentials + full URL
  const { data: prs, isLoading, error, isRefetching } = useQuery<PullRequest[]>({
    queryKey: ['/api/pull-requests'],
    queryFn: async () => {
      const res = await fetch("http://localhost:5000/api/pull-requests", {
        credentials: "include",
      });
      if (!res.ok) throw new Error("Failed to fetch pull requests");
      return res.json();
    }
  });

  const handleFilterToggle = (id: string) => {
    setFilters(filters.map(f => 
      f.id === id ? { ...f, active: !f.active } : f
    ));
  };

  const handleClearFilters = () => {
    setFilters(filters.map(f => ({ ...f, active: false })));
  };

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['/api/pull-requests'] });
  };

  const filteredPRs = useMemo(() => {
    if (!prs) return [];

    const activeFilters = filters.filter(f => f.active).map(f => f.id);
    
    return prs.filter(pr => {
      const matchesSearch = 
        pr.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        pr.repository.toLowerCase().includes(searchQuery.toLowerCase()) ||
        pr.user.login.toLowerCase().includes(searchQuery.toLowerCase());
      
      let prStatus = pr.state;
      if (pr.merged) {
        prStatus = 'merged';
      }
      
      const matchesFilter = activeFilters.length === 0 || activeFilters.includes(prStatus);
      
      return matchesSearch && matchesFilter;
    });
  }, [prs, searchQuery, filters]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto">
        <div className="max-w-5xl mx-auto p-6 space-y-6">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <h1 className="text-2xl font-semibold text-foreground">Pull Requests</h1>
              <p className="text-sm text-muted-foreground mt-1">
                All pull requests across your repositories
              </p>
            </div>
            <Button
              variant="outline"
              onClick={handleRefresh}
              disabled={isRefetching}
              data-testid="button-refresh-prs"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefetching ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>

          <div className="flex flex-col gap-4">
            <SearchBar
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search pull requests..."
            />
            <FilterBar
              filters={filters}
              onFilterToggle={handleFilterToggle}
              onClearAll={handleClearFilters}
            />
          </div>

          {error ? (
            <ErrorState
              title="Failed to load pull requests"
              message="Unable to fetch pull requests from GitHub. Please check your connection and try again."
              onRetry={handleRefresh}
            />
          ) : isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-40" />
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              {filteredPRs.map((pr) => (
                <PullRequestCard
                  key={pr.id}
                  number={pr.number}
                  title={pr.title}
                  author={{
                    name: pr.user.login,
                    avatar: pr.user.avatar_url,
                  }}
                  status={(pr.merged ? 'merged' : pr.state) as "open" | "merged" | "closed"}
                  createdAt={new Date(pr.created_at)}
                  repository={pr.repository}
                  url={pr.html_url}
                />
              ))}
            </div>
          )}

          {!isLoading && !error && filteredPRs.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">
                {searchQuery || filters.some(f => f.active) 
                  ? "No pull requests found" 
                  : "No pull requests available"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
