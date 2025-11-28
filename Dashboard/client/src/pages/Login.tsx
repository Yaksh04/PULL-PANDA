// // SWE_project_website/client/src/pages/Login.tsx
// export default function Login() {
//   const handleLogin = () => {
//     window.location.href = "http://localhost:5000/api/auth/github";
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

// src/pages/Login.tsx

import logo from "./logo.jpg";

export default function Login() {
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

  const handleLogin = () => {
    window.location.href = `${API_URL}/api/auth/github`;
  };

  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen bg-black overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden opacity-20">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-gray-800 rounded-full mix-blend-overlay blur-3xl animate-pulse"></div>
        <div
          className="absolute top-1/3 right-1/4 w-96 h-96 bg-gray-700 rounded-full mix-blend-overlay blur-3xl animate-pulse"
          style={{ animationDelay: "1s" }}
        ></div>
        <div
          className="absolute bottom-1/4 left-1/3 w-96 h-96 bg-gray-800 rounded-full mix-blend-overlay blur-3xl animate-pulse"
          style={{ animationDelay: "2s" }}
        ></div>
      </div>

      <div className="relative z-10 flex flex-col items-center">
        <div className="mb-12 transform hover:scale-105 transition-transform duration-300">
          <div className="bg-white rounded-3xl p-2 shadow-2xl shadow-gray-800">
            <img src={logo} alt="PR Review Agent Logo" className="w-24 h-24" />
          </div>
        </div>

        <h1 className="text-5xl font-bold mb-3 text-white text-center tracking-tight">
          PR Review Agent
        </h1>
        <p className="text-gray-400 text-lg mb-12 text-center max-w-md">
          Sign in with your GitHub account to continue
        </p>

        <button
          className="group relative px-8 py-3 bg-white text-black rounded-lg font-medium text-base shadow-lg hover:bg-gray-200 transition-all duration-200 overflow-hidden"
          onClick={handleLogin}
        >
          <span className="relative flex items-center gap-3">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
              />
            </svg>
            Continue with GitHub
          </span>
        </button>
      </div>
    </div>
  );
}
