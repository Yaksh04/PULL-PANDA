// SWE_project_website/client/src/App.tsx

import { Switch, Route, useLocation } from "wouter";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";

import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/ThemeProvider";
import { Toaster } from "@/components/ui/toaster";

// UI Components (Restored)
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { ThemeToggle } from "@/components/ThemeToggle";

// Pages
import Dashboard from "@/pages/Dashboard";
import PullRequests from "@/pages/PullRequests";
import Reviews from "@/pages/Reviews";
import Analytics from "@/pages/Analytics";
import PRDetails from "@/pages/PRDetails";
import NotFound from "@/pages/not-found";
import Login from "@/pages/Login";

import ProtectedRoute from "./ProtectedRoute";
import "./index.css";

import { useEffect, useState } from "react";

export default function App() {
  const [location, setLocation] = useLocation();
  const [isAuthReady, setIsAuthReady] = useState(false);

  // 1. Token Check Logic (From previous fix)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");

    if (token) {
      console.log("Token found in URL. Saving to storage...");
      localStorage.setItem("github_token", token);
      window.history.replaceState({}, document.title, window.location.pathname);

      if (window.location.pathname === "/login") {
        setLocation("/");
      }
    }
    setIsAuthReady(true);
  }, [setLocation]);

  // 2. Loading Screen
  if (!isAuthReady) {
    return (
      <div className="flex h-screen items-center justify-center">
        Loading...
      </div>
    );
  }

  // 3. Check if we are on the Login page
  const isLoginPage = location === "/login";

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <ThemeProvider defaultTheme="dark">
          {/* CONDITIONAL LAYOUT */}
          {isLoginPage ? (
            // Scenario A: Login Page (Full Screen, No Sidebar)
            <Switch>
              <Route path="/login" component={Login} />
            </Switch>
          ) : (
            // Scenario B: App Layout (Sidebar + Header + Content)
            <SidebarProvider>
              <div className="flex h-screen w-full">
                {/* THE SIDEBAR */}
                <AppSidebar />

                <div className="flex flex-col flex-1 min-w-0">
                  {/* THE HEADER (Trigger + Theme Toggle) */}
                  <header className="flex items-center justify-between px-6 py-3 border-b border-border shrink-0">
                    <SidebarTrigger />
                    <ThemeToggle />
                  </header>

                  {/* THE PAGE CONTENT */}
                  <main className="flex-1 overflow-hidden overflow-y-auto">
                    <Switch>
                      <Route
                        path="/"
                        component={() => (
                          <ProtectedRoute>
                            <Dashboard />
                          </ProtectedRoute>
                        )}
                      />

                      <Route
                        path="/pull-requests"
                        component={() => (
                          <ProtectedRoute>
                            <PullRequests />
                          </ProtectedRoute>
                        )}
                      />

                      <Route
                        path="/pr-details/:owner/:repo/:number"
                        component={() => (
                          <ProtectedRoute>
                            <PRDetails />
                          </ProtectedRoute>
                        )}
                      />

                      <Route
                        path="/reviews"
                        component={() => (
                          <ProtectedRoute>
                            <Reviews />
                          </ProtectedRoute>
                        )}
                      />

                      <Route
                        path="/analytics"
                        component={() => (
                          <ProtectedRoute>
                            <Analytics />
                          </ProtectedRoute>
                        )}
                      />

                      {/* Handle 404 inside the layout */}
                      <Route component={NotFound} />
                    </Switch>
                  </main>
                </div>
              </div>
            </SidebarProvider>
          )}

          <Toaster />
        </ThemeProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
