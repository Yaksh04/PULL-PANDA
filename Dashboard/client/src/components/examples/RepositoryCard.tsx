import { RepositoryCard } from "../RepositoryCard";

export default function RepositoryCardExample() {
  return (
    <RepositoryCard
      name="ai-review-agent"
      description="Automated PR review system using GPT-4 and static analysis tools"
      stars={234}
      forks={45}
      openPRs={3}
      language="TypeScript"
      isPrivate={false}
      onClick={() => console.log("Repository clicked")}
    />
  );
}
