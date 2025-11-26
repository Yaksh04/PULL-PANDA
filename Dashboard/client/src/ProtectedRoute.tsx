// // SWE_project_website\client\src\ProtectedRoute.tsx
// import { Redirect } from "wouter";
// import { useQuery } from "@tanstack/react-query";

// interface ProtectedProps {
//   component: React.ComponentType<any>;
// }

// export default function ProtectedRoute({ component: Component }: ProtectedProps) {
//   const { data, isLoading, error } = useQuery({
//     queryKey: ["/api/auth/me"],
//     queryFn: async () => {
//       const res = await fetch("/api/auth/me", {
//         credentials: "include"
//       });
//       if (!res.ok) throw new Error("Not authenticated");
//       return res.json();
//     },
//     retry: false,
//   });

//   if (isLoading) return <div>Loading…</div>;

//   if (error) return <Redirect to="/login" />;

//   return <Component user={data} />;
// }

// client/src/components/ProtectedRoute.tsx
import { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/apiClient";
import { Redirect } from "wouter";

interface Props {
  children: ReactNode;
}

export default function ProtectedRoute({ children }: Props) {
  const { isLoading, error } = useQuery({
    queryKey: ["auth-check"], // 🔥 DIFFERENT from Navbar
    queryFn: () => apiFetch("/api/auth/me"), // fresh check
    retry: false,
  });

  // Still checking login session
  if (isLoading) return <div className="text-white p-8">Loading...</div>;

  // User not logged in
  if (error) return <Redirect to="/login" />;

  // User logged in
  return <>{children}</>;
}
