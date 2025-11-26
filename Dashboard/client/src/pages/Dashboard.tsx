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
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477..." />
            </svg>
            Continue with GitHub
          </span>
        </button>
      </div>
    </div>
  );
}
