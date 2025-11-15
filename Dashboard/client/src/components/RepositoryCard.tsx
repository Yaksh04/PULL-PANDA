import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Star, GitFork, GitPullRequest } from "lucide-react";

interface RepositoryCardProps {
  name: string;
  description: string;
  stars: number;
  forks: number;
  openPRs: number;
  language?: string;
  isPrivate: boolean;
  onClick: () => void;
}

export function RepositoryCard({
  name,
  description,
  stars,
  forks,
  openPRs,
  language,
  isPrivate,
  onClick,
}: RepositoryCardProps) {
  return (
    <Card
      onClick={onClick}
      className="p-4 cursor-pointer hover:bg-accent transition"
    >
      <div className="flex justify-between items-start">
        <h3 className="text-lg font-semibold">{name}</h3>
        {isPrivate && <Badge variant="destructive">Private</Badge>}
      </div>

      <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
        {description}
      </p>

      <div className="flex items-center gap-4 mt-3 text-sm text-muted-foreground">
        <span className="flex items-center gap-1">
          <Star className="h-4 w-4" /> {stars}
        </span>

        <span className="flex items-center gap-1">
          <GitFork className="h-4 w-4" /> {forks}
        </span>

        <span className="flex items-center gap-1">
          <GitPullRequest className="h-4 w-4" /> {openPRs}
        </span>

        {language && (
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-blue-400" />
            {language}
          </span>
        )}
      </div>
    </Card>
  );
}
