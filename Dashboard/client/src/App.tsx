// import { Switch, Route } from "wouter";
// import { queryClient } from "./lib/queryClient";
// import { QueryClientProvider, useQuery } from "@tanstack/react-query";
// import { Toaster } from "@/components/ui/toaster";
// import { TooltipProvider } from "@/components/ui/tooltip";
// import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
// import { AppSidebar } from "@/components/AppSidebar";
// import { ThemeProvider } from "@/components/ThemeProvider";
// import { ThemeToggle } from "@/components/ThemeToggle";
// import { SiGithub } from "react-icons/si";
// import Dashboard from "@/pages/Dashboard";
// import PullRequests from "@/pages/PullRequests";
// import Reviews from "@/pages/Reviews";
// import Analytics from "@/pages/Analytics";
// import NotFound from "@/pages/not-found";
// import { useEffect, useState } from "react";

// type GitHubUser = {
//   login: string;
//   avatar_url: string;
// };

// function GitHubUserInfo() {
//   const { data: user } = useQuery<GitHubUser>({
//     queryKey: ["/api/user"],
//     retry: false,
//   });

//   if (!user) return null;

//   return (
//     <div className="flex items-center gap-2 px-4 py-2 border-b border-border">
//       <SiGithub className="h-4 w-4 text-muted-foreground" />
//       <span className="text-sm text-muted-foreground">
//         Connected as{" "}
//         <span className="font-medium text-foreground">{user.login}</span>
//       </span>
//     </div>
//   );
// }

// function Router({ authenticated }: { authenticated: boolean }) {
//   if (!authenticated) {
//     return null;
//   }

//   return (
//     <Switch>
//       <Route path="/" component={Dashboard} />
//       <Route path="/pull-requests" component={PullRequests} />
//       <Route path="/reviews" component={Reviews} />
//       <Route path="/analytics" component={Analytics} />
//       <Route component={NotFound} />
//     </Switch>
//   );
// }

// // ✅ FIX STARTS HERE
// export default function App() {
//   const style = {
//     "--sidebar-width": "15rem",
//     "--sidebar-width-icon": "3rem",
//   } as React.CSSProperties;

//   const [authenticated, setAuthenticated] = useState<boolean>(false);
//   const [loading, setLoading] = useState<boolean>(true);

//   useEffect(() => {
//     fetch("/api/auth/me")
//       .then((res) => {
//         if (res.status === 200) {
//           setAuthenticated(true);
//         } else {
//           window.location.replace("/api/auth/github");
//         }
//       })
//       .catch(() => {
//         window.location.replace("/api/auth/github");
//       })
//       .finally(() => setLoading(false));
//   }, []);

//   if (loading) return null;

//   return (
//     <QueryClientProvider client={queryClient}>
//       <TooltipProvider>
//         <ThemeProvider defaultTheme="dark">
//           <SidebarProvider style={style}>
//             <div className="flex h-screen w-full">
//               <AppSidebar />
//               <div className="flex flex-col flex-1 min-w-0">
//                 <header className="flex items-center justify-between px-6 py-3 border-b border-border shrink-0">
//                   <SidebarTrigger data-testid="button-sidebar-toggle" />
//                   <ThemeToggle />
//                 </header>
//                 <GitHubUserInfo />
//                 <main className="flex-1 overflow-hidden">
//                   <Router authenticated={authenticated} />
//                 </main>
//               </div>
//             </div>
//           </SidebarProvider>
//           <Toaster />
//         </ThemeProvider>
//       </TooltipProvider>
//     </QueryClientProvider>
//   );
// }

// SWE_project_website/client/src/App.tsx
import { Switch, Route } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";

import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/ThemeProvider";
import { Toaster } from "@/components/ui/toaster";

import Dashboard from "@/pages/Dashboard";
import PullRequests from "@/pages/PullRequests";
import Reviews from "@/pages/Reviews";
import Analytics from "@/pages/Analytics";
import NotFound from "@/pages/not-found";
import Login from "@/pages/Login";

import ProtectedRoute from "./ProtectedRoute";

import "./index.css";

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <ThemeProvider defaultTheme="dark">
          <Switch>

            {/* PUBLIC LOGIN ROUTE */}
            <Route path="/login">
              <Login />
            </Route>

            {/* PROTECTED ROUTES */}
            <Route path="/">
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            </Route>

            <Route path="/pull-requests">
              <ProtectedRoute>
                <PullRequests />
              </ProtectedRoute>
            </Route>

            <Route path="/reviews">
              <ProtectedRoute>
                <Reviews />
              </ProtectedRoute>
            </Route>

            <Route path="/analytics">
              <ProtectedRoute>
                <Analytics />
              </ProtectedRoute>
            </Route>

            {/* 404 */}
            <Route>
              <NotFound />
            </Route>

          </Switch>

          <Toaster />
        </ThemeProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
