# AI GitHub PR Review Agent - Design Guidelines

## Design Approach

**Selected Approach:** Reference-Based (GitHub + Linear.app fusion)

This developer dashboard requires the precision and familiarity of established developer tools. Drawing primary inspiration from GitHub's interface patterns and Linear's clean typography, with focus on data density and utility over visual flourish.

## Core Design Principles

1. **Developer-First Clarity:** Information hierarchy optimized for scanning large lists of PRs
2. **Familiar Patterns:** Leverage GitHub's established UI conventions for zero learning curve
3. **Data Density:** Maximize information per viewport without overwhelming
4. **Speed Perception:** Instant visual feedback, skeleton states during loading

## Color System

**Dark Mode (Primary):**
- Background: #0d1117 (GitHub dark)
- Surface: #161b22 (cards, sidebar)
- Text: #f0f6fc (primary), #8b949e (secondary)
- Accent: #238636 (success/merged), #da3633 (closed), #1f6feb (links/primary actions)
- Borders: #30363d

**Light Mode:**
- Background: #ffffff
- Surface: #f6f8fa
- Text: #24292f (primary), #57606a (secondary)
- Accent colors remain consistent
- Borders: #d0d7de

## Typography

**Fonts:**
- Primary: 'Inter' (body, UI elements) via Google Fonts
- Monospace: 'JetBrains Mono' (code snippets, commit hashes)

**Hierarchy:**
- Page titles: text-2xl, font-semibold
- Section headers: text-lg, font-medium
- PR titles: text-base, font-medium
- Metadata (dates, authors): text-sm, text-secondary
- Code/hashes: text-xs, font-mono

## Layout System

**Spacing Units:** Tailwind scale of 2, 4, 6, 8, 12, 16
- Component padding: p-4 to p-6
- Section gaps: gap-6 to gap-8
- Card spacing: space-y-4

**Grid Structure:**
- Sidebar: Fixed 240px width (w-60)
- Main content: flex-1 with max-w-7xl container
- PR list: Single column, full-width cards
- Stats grid: grid-cols-3 or grid-cols-4 for metrics

## Component Library

### Navigation
**Sidebar:**
- Fixed left sidebar with logo at top
- Navigation items with icons (from Heroicons)
- Active state: subtle background highlight + accent border-left
- Sections: My Repos, Pull Requests, Reviews, Analytics
- User profile at bottom with theme toggle

**Top Bar:**
- Search/filter controls (left)
- Refresh button (right)
- Breadcrumb navigation when drilling into repos

### Repository Cards
- Compact cards showing: repo name, description, PR count badge
- Status indicators: open PRs (blue dot), recent activity
- Click to expand/navigate to PR list
- Grid layout on larger screens (2-3 cols)

### Pull Request List Items
**Each PR card contains:**
- PR title (prominent, truncated with tooltip)
- Metadata row: #PR number, author avatar + name, time ago
- Status badge: Open (blue), Merged (purple), Closed (red)
- AI Review summary: collapsible section with review sentiment icon
- Static Analysis: chips showing issue count by severity
- Action buttons: View on GitHub (external link icon)

**Layout:**
- Border-left accent color by status
- Hover state: subtle background change
- Expandable sections for full AI comments

### Summary Statistics Dashboard
**Metrics Cards:**
- Grid of 3-4 key metrics at top
- Large number (text-4xl), label below (text-sm)
- Subtle icon (Heroicons outline)
- Cards: PRs Reviewed, Acceptance Rate, Avg Review Time, Active Repos

**Charts Section:**
- Simple bar chart: PRs per week (placeholder for future)
- Status breakdown: donut chart visualization
- Keep minimal, avoid heavy chart libraries initially

### Filters & Search
- Horizontal filter bar below top nav
- Status filters: pill-style toggle buttons
- Search input with icon, placeholder "Search PRs..."
- Clear all filters button
- Active filters shown as removable chips

### Loading & Empty States
- Skeleton loaders matching card structure
- Empty state: centered icon + message + action (e.g., "Connect repos")
- Error state: friendly message with retry action

## Interactions

**Minimal Animations:**
- Hover states: opacity/background transitions (150ms)
- Expand/collapse: smooth height transition (200ms)
- Page transitions: fade (100ms)
- NO complex scroll animations or parallax

**Micro-interactions:**
- Refresh button: rotate icon during fetch
- PR status changes: subtle flash highlight
- Filter application: instant feedback

## Responsive Behavior

**Desktop (lg+):** Full sidebar + multi-column grids
**Tablet (md):** Collapsible sidebar, single-column PR list
**Mobile:** Hidden sidebar (hamburger menu), stacked stats

## Images

**No hero images or marketing imagery** - this is a data dashboard, not a landing page. Focus on:
- User avatars (GitHub profile pics)
- Repository icons/logos where applicable
- Status icons throughout

## Accessibility

- All interactive elements keyboard navigable
- ARIA labels on icon-only buttons
- Focus indicators matching GitHub style (blue outline)
- Sufficient contrast ratios in both themes (WCAG AA)

## Key Differentiators

- **No marketing fluff:** Pure utility interface
- **GitHub familiarity:** Users feel at home immediately
- **Information first:** Dense but scannable layouts
- **Performance signals:** Loading states, optimistic updates
- **Professional restraint:** No unnecessary animations or visual tricks

This dashboard prioritizes developer productivity over visual spectacle - every pixel serves a functional purpose.