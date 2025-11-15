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


// SWE_project_website/client/src/ProtectedRoute.tsx
import { Redirect } from "wouter";
import { useQuery } from "@tanstack/react-query";

interface Props {
  children: React.ReactNode;
}

export default function ProtectedRoute({ children }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["auth"],
    queryFn: async () => {
      const res = await fetch("http://localhost:5000/api/auth/me", {
        credentials: "include"
      });
      if (!res.ok) throw new Error("Not authenticated");
      return res.json();
    },
    retry: false,
  });

  if (isLoading) return <div>Loading...</div>;

  if (error) return <Redirect to="/login" />;

  return <>{children}</>;
}
