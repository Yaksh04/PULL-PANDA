import { useState } from "react";
import { FilterBar } from "../FilterBar";

export default function FilterBarExample() {
  const [filters, setFilters] = useState([
    { id: "open", label: "Open", active: true },
    { id: "merged", label: "Merged", active: false },
    { id: "closed", label: "Closed", active: false },
  ]);

  const handleFilterToggle = (id: string) => {
    setFilters(filters.map(f => 
      f.id === id ? { ...f, active: !f.active } : f
    ));
  };

  const handleClearFilters = () => {
    setFilters(filters.map(f => ({ ...f, active: false })));
  };

  return (
    <FilterBar
      filters={filters}
      onFilterToggle={handleFilterToggle}
      onClearAll={handleClearFilters}
    />
  );
}
