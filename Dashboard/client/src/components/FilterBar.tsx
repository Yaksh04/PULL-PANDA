import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";

interface FilterOption {
  id: string;
  label: string;
  active: boolean;
}

interface FilterBarProps {
  filters: FilterOption[];
  onFilterToggle: (id: string) => void;
  onClearAll: () => void;
}

export function FilterBar({ filters, onFilterToggle, onClearAll }: FilterBarProps) {
  const activeCount = filters.filter(f => f.active).length;

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-sm text-muted-foreground">Filter:</span>
      {filters.map((filter) => (
        <Button
          key={filter.id}
          variant={filter.active ? "default" : "outline"}
          size="sm"
          onClick={() => onFilterToggle(filter.id)}
          className="h-8"
          data-testid={`button-filter-${filter.id}`}
        >
          {filter.label}
        </Button>
      ))}
      {activeCount > 0 && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onClearAll}
          className="h-8 text-muted-foreground"
          data-testid="button-clear-filters"
        >
          <X className="h-3 w-3 mr-1" />
          Clear all
        </Button>
      )}
    </div>
  );
}
