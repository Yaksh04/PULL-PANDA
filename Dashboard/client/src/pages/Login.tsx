// import React from "react";

// const GITHUB_CLIENT_ID = process.env.REACT_APP_GITHUB_CLIENT_ID || "YOUR_CLIENT_ID";
// const REDIRECT_URI = process.env.REACT_APP_GITHUB_REDIRECT_URI || "http://localhost:3000/api/auth/github/callback";

// export default function Login() {
//   const handleLogin = () => {
//     window.location.href = `https://github.com/login/oauth/authorize?client_id=${GITHUB_CLIENT_ID}&redirect_uri=${encodeURIComponent(REDIRECT_URI)}`;
//   };

//   return (
//     <div className="flex flex-col items-center justify-center h-screen">
//       <h1 className="text-3xl font-bold mb-6">Sign in with GitHub</h1>
//       <button
//         className="bg-black text-white px-6 py-2 rounded hover:bg-gray-800"
//         onClick={handleLogin}
//       >
//         Login with GitHub
//       </button>
//     </div>
//   );
// }

// SWE_project_website/client/src/pages/Login.tsx
export default function Login() {
  const handleLogin = () => {
    window.location.href = "http://localhost:5000/api/auth/github";
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <h1 className="text-3xl font-bold mb-6">Sign in with GitHub</h1>
      <button
        className="bg-black text-white px-6 py-2 rounded hover:bg-gray-800"
        onClick={handleLogin}
      >
        Login with GitHub
      </button>
    </div>
  );
}
