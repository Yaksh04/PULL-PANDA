export function getToken() {
  return localStorage.getItem("github_token");
}

export function setToken(token: string) {
  localStorage.setItem("github_token", token);
}

export function clearToken() {
  localStorage.removeItem("github_token");
}

export function isAuthenticated() {
  return !!getToken();
}
