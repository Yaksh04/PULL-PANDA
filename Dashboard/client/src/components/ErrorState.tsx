import { AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ 
  title = "Something went wrong", 
  message, 
  onRetry 
}: ErrorStateProps) {
  return (
    <Card className="p-8 text-center">
      <div className="flex flex-col items-center gap-4">
        <div className="h-12 w-12 rounded-full bg-destructive/10 flex items-center justify-center">
          <AlertCircle className="h-6 w-6 text-destructive" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-card-foreground mb-1">
            {title}
          </h3>
          <p className="text-sm text-muted-foreground max-w-md">
            {message}
          </p>
        </div>
        {onRetry && (
          <Button onClick={onRetry} variant="outline" data-testid="button-retry">
            Try Again
          </Button>
        )}
      </div>
    </Card>
  );
}
