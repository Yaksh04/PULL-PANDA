import { StatsCard } from "../StatsCard";
import { GitPullRequest } from "lucide-react";

export default function StatsCardExample() {
  return (
    <StatsCard
      title="Total PRs Reviewed"
      value="127"
      icon={GitPullRequest}
      trend="+12 this week"
    />
  );
}
