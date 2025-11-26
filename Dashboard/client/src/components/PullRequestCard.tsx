import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  ExternalLink,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  CheckCircle2,
  ThumbsUp,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useLocation } from "wouter";

interface PullRequestCardProps {
  id?: number;
  number: number;
  title: string;
  author: {
    name: string;
    avatar: string;
  };
  status: "open" | "merged" | "closed";
  createdAt: Date;
  repository: string;
  owner: string;
  url: string;

  aiReviewed?: boolean;

  aiReview?: {
    sentiment: "approved" | "changes_requested" | "commented";
    summary: string;
  };

  staticAnalysis?: {
    issues: number;
    warnings: number;
  };

  // Optional: Allow parent to override click
  onClick?: () => void;
}

const statusConfig = {
  open: { color: "bg-chart-1", label: "Open", variant: "default" as const },
  merged: {
    color: "bg-chart-3",
    label: "Merged",
    variant: "secondary" as const,
  },
  closed: {
    color: "bg-destructive",
    label: "Closed",
    variant: "destructive" as const,
  },
};

const sentimentIcons = {
  approved: CheckCircle2,
  changes_requested: AlertCircle,
  commented: ThumbsUp,
};

export function PullRequestCard(props: PullRequestCardProps) {
  const {
    number,
    title,
    author,
    status,
    createdAt,
    repository,
    owner,
    url,
    aiReviewed,
    aiReview,
    staticAnalysis,
    onClick, // Grab the prop if passed
  } = props;

  const [expanded, setExpanded] = useState(false);
  const [, setLocation] = useLocation();

  const statusInfo = statusConfig[status];
  const SentimentIcon = aiReview ? sentimentIcons[aiReview.sentiment] : null;

  const handleCardClick = () => {
    if (onClick) {
      onClick();
    } else {
      // FIX: Use the correct dynamic route format
      setLocation(`/pr-details/${owner}/${repository}/${number}`);
    }
  };

  return (
    <Card
      className="p-4 cursor-pointer hover:bg-accent transition"
      onClick={handleCardClick}
      data-testid={`card-pr-${number}`}
    >
      <div className="flex flex-col gap-3">
        <div className="flex items-start gap-3">
          {/* AI REVIEW DOT */}
          <div className="flex flex-col items-center pt-2">
            <div
              className={`h-2 w-2 rounded-full ${
                aiReviewed ? "bg-green-500" : "bg-orange-400"
              }`}
              title={aiReviewed ? "AI Reviewed" : "Pending AI Review"}
            />
          </div>

          {/* Original status dot */}
          <div
            className={`h-1 w-1 rounded-full ${statusInfo.color} mt-2 shrink-0`}
          />

          {/* MAIN CONTENT */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2 mb-2">
              <div className="flex-1 min-w-0">
                <h3 className="text-base font-medium text-card-foreground mb-1">
                  {title}
                </h3>

                <div className="flex items-center gap-2 flex-wrap text-sm text-muted-foreground">
                  <span className="font-mono">#{number}</span>
                  <span>•</span>
                  <span>{repository}</span>
                  <span>•</span>

                  <div className="flex items-center gap-1">
                    <Avatar className="h-4 w-4">
                      <AvatarImage src={author.avatar} />
                      <AvatarFallback>{author.name[0]}</AvatarFallback>
                    </Avatar>
                    <span>{author.name}</span>
                  </div>

                  <span>•</span>
                  <span>
                    {formatDistanceToNow(createdAt, { addSuffix: true })}
                  </span>
                </div>
              </div>

              {/* Status Badge */}
              <Badge variant={statusInfo.variant} className="shrink-0">
                {statusInfo.label}
              </Badge>
            </div>

            {/* AI Review Sentiment + Static Analysis */}
            {(aiReview || staticAnalysis) && (
              <div className="flex items-center gap-2 mb-2">
                {aiReview && SentimentIcon && (
                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <SentimentIcon className="h-3.5 w-3.5" />
                    <span className="capitalize">
                      {aiReview.sentiment.replace("_", " ")}
                    </span>
                  </div>
                )}

                {staticAnalysis && (
                  <>
                    {staticAnalysis.issues > 0 && (
                      <Badge variant="destructive" className="text-xs">
                        {staticAnalysis.issues} issue
                        {staticAnalysis.issues !== 1 ? "s" : ""}
                      </Badge>
                    )}
                    {staticAnalysis.warnings > 0 && (
                      <Badge variant="secondary" className="text-xs">
                        {staticAnalysis.warnings} warning
                        {staticAnalysis.warnings !== 1 ? "s" : ""}
                      </Badge>
                    )}
                  </>
                )}
              </div>
            )}

            {/* Expandable AI Review Summary */}
            {aiReview && (
              <div
                className="mt-2"
                onClick={(e) => e.stopPropagation()} // prevent whole card click
              >
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setExpanded(!expanded)}
                  className="h-7 px-2 text-xs"
                  data-testid={`button-expand-review-${number}`}
                >
                  {expanded ? (
                    <>
                      <ChevronUp className="h-3 w-3 mr-1" />
                      Hide AI Review
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-3 w-3 mr-1" />
                      Show AI Review
                    </>
                  )}
                </Button>

                {expanded && (
                  <div className="mt-2 p-3 bg-muted rounded-md text-sm text-muted-foreground">
                    {aiReview.summary}
                  </div>
                )}
              </div>
            )}

            {/* GitHub Button */}
            <div
              className="flex items-center gap-2 mt-3"
              onClick={(e) => e.stopPropagation()}
            >
              <Button variant="outline" size="sm" asChild>
                <a href={url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-3 w-3 mr-1" />
                  View on GitHub
                </a>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
