import { PullRequestCard } from "../PullRequestCard";

export default function PullRequestCardExample() {
  return (
    <PullRequestCard
      number={142}
      title="Add authentication middleware for API endpoints"
      author={{ name: "sarah-dev", avatar: "https://github.com/github.png" }}
      status="open"
      createdAt={new Date(Date.now() - 2 * 60 * 60 * 1000)}
      repository="backend-services"
      aiReview={{
        sentiment: "approved",
        summary: "Code looks good! Authentication implementation follows best practices. JWT token handling is secure.",
      }}
      staticAnalysis={{
        issues: 0,
        warnings: 2,
      }}
      url="https://github.com/example/backend-services/pull/142"
    />
  );
}
