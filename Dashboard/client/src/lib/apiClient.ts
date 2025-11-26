const API_URL = import.meta.env.VITE_API_URL || ""; // Ensure this is set correctly in .env

export async function apiFetch(path: string, options: RequestInit = {}) {
  // 1. Grab the token from storage
  const token = localStorage.getItem("github_token");

  // 2. Create headers object
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {}),
  };

  // 3. If token exists, attach it as Bearer token
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}${path}`, {
    ...options,

    headers, // Inject the headers with the token
  });

  if (!res.ok) {
    // Optional: If 401 (Unauthorized), redirect to login
    if (res.status === 401) {
      window.location.href = "/login";
    }
    throw new Error(`API error: ${res.status}`);
  }

  return res.json();
}
