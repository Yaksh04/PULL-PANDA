// // SWE_project_website/server/auth.ts
// import dotenv from "dotenv";
// dotenv.config(); // MUST be first

// import express, { Request, Response } from "express";
// import axios from "axios";
// import { Octokit } from "@octokit/rest";
// import session from "express-session";

// declare module "express-session" {
//   interface SessionData {
//     accessToken?: string;
//   }
// }

// const router = express.Router();

// console.log("AUTH ENV TEST:", process.env.GITHUB_CLIENT_ID, process.env.GITHUB_REDIRECT_URI);

// /* ------------------------------------------------------
//    STEP 1 — LOGIN ROUTE
//    Forces GitHub to prompt login + consent every time.
// -------------------------------------------------------- */
// router.get("/github", (_req: Request, res: Response) => {
//   const CLIENT_ID = process.env.GITHUB_CLIENT_ID;
//   const REDIRECT_URI = process.env.GITHUB_REDIRECT_URI;

//   if (!CLIENT_ID || !REDIRECT_URI) {
//     return res.status(500).send("GitHub OAuth is not configured.");
//   }

//   const authUrl =
//     `https://github.com/login/oauth/authorize` +
//     `?client_id=${CLIENT_ID}` +
//     `&redirect_uri=${encodeURIComponent(REDIRECT_URI)}` +
//     `&scope=repo,user` +
//     `&prompt=consent` +
//     `&force_verify=true`;  // <--- THE KEY FIX

//   res.redirect(authUrl);
// });

// /* ------------------------------------------------------
//    STEP 2 — CALLBACK ROUTE
//    GitHub sends ?code=XYZ → we exchange it for access token
// -------------------------------------------------------- */
// router.get("/github/callback", async (req: Request, res: Response) => {
//   const CLIENT_ID = process.env.GITHUB_CLIENT_ID;
//   const CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET;
//   const REDIRECT_URI = process.env.GITHUB_REDIRECT_URI;
//   const FRONTEND_URL = process.env.FRONTEND_URL;

//   if (!CLIENT_ID || !CLIENT_SECRET || !REDIRECT_URI || !FRONTEND_URL) {
//     console.error("❌ Missing OAuth ENV vars in callback");
//     return res.status(500).send("OAuth configuration incomplete.");
//   }

//   const code = req.query.code as string;

//   if (!code) {
//     console.error("❌ OAuth Callback Error: Missing code");
//     return res.status(400).send("Missing 'code' parameter from GitHub OAuth.");
//   }

//   try {
//     const tokenResponse = await axios.post(
//       "https://github.com/login/oauth/access_token",
//       {
//         client_id: CLIENT_ID,
//         client_secret: CLIENT_SECRET,
//         code,
//         redirect_uri: REDIRECT_URI
//       },
//       { headers: { Accept: "application/json" } }
//     );

//     const accessToken = tokenResponse.data.access_token;

//     if (!accessToken) {
//       console.error("❌ Failed to exchange OAuth code:", tokenResponse.data);
//       return res.status(401).send("GitHub OAuth token exchange failed.");
//     }

//     // Save token in session
//     req.session.accessToken = accessToken;

//     console.log("✔ OAuth success — redirecting to frontend:", FRONTEND_URL);
//     return res.redirect(FRONTEND_URL);

//   } catch (err) {
//     console.error("❌ OAuth callback exchange failed:", err);
//     return res.status(500).send("GitHub OAuth failed during token exchange.");
//   }
// });

// /* ------------------------------------------------------
//    STEP 3 — CHECK AUTH STATE
//    Used by ProtectedRoute (frontend)
// -------------------------------------------------------- */
// router.get("/me", async (req: Request, res: Response) => {
//   if (!req.session.accessToken) {
//     return res.status(401).json({ error: "Not authenticated" });
//   }

//   try {
//     const octokit = new Octokit({ auth: req.session.accessToken });
//     const { data: user } = await octokit.rest.users.getAuthenticated();
//     return res.json(user);

//   } catch (err) {
//     console.error("❌ Token invalid — clearing session:", err);
//     delete req.session.accessToken;
//     return res.status(401).json({ error: "Token invalid, please log in again." });
//   }
// });

// /* ------------------------------------------------------
//    STEP 4 — LOGOUT
// -------------------------------------------------------- */
// router.post("/logout", (req: Request, res: Response) => {
//   req.session.destroy(() => {
//     res.clearCookie("connect.sid", {
//       path: "/",
//       sameSite: "none",
//       secure: false
//     });
//     console.log("✔ Logged out successfully.");
//     return res.json({ message: "Logged out" });
//   });
// });

// export default router;

// SWE_project_website/server/auth.ts

import express from "express";
import axios from "axios";
import { Octokit } from "@octokit/rest";

const router = express.Router();

/* ENVIRONMENT VARIABLES*/

// 1. Backend URL: Railway provides this automatically as RAILWAY_PUBLIC_DOMAIN
const BACKEND_URL = process.env.RAILWAY_PUBLIC_DOMAIN
  ? `https://${process.env.RAILWAY_PUBLIC_DOMAIN}`
  : process.env.BACKEND_URL || "http://localhost:5000";

// 2. Frontend URL: You set this manually in Railway variables
const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:3000";

// 3. GitHub Keys: You set these manually in Railway variables
const GITHUB_CLIENT_ID = process.env.GITHUB_CLIENT_ID;
const GITHUB_CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET;

// Safety Check: Crash if keys are missing (helps debugging)
if (!GITHUB_CLIENT_ID || !GITHUB_CLIENT_SECRET) {
  console.error("CRITICAL ERROR: Missing GitHub OAuth Credentials in .env");
  throw new Error("Missing GitHub OAuth Credentials");
}

/* 
   STEP 1 — LOGIN → REDIRECT TO GITHUB
 */
router.get("/github", (_req, res) => {
  const redirectUri = `${BACKEND_URL}/api/auth/github/callback`;

  const url =
    `https://github.com/login/oauth/authorize` +
    `?client_id=${GITHUB_CLIENT_ID}` +
    `&redirect_uri=${encodeURIComponent(redirectUri)}` +
    `&scope=repo,user` +
    `&prompt=consent&force_verify=true`;

  res.redirect(url);
});

/* 
   STEP 2 — GITHUB CALLBACK
 */
router.get("/github/callback", async (req, res) => {
  const code = req.query.code as string;

  if (!code) {
    return res.status(400).send("Missing OAuth code.");
  }

  try {
    const redirectUri = `${BACKEND_URL}/api/auth/github/callback`;

    const tokenRes = await axios.post(
      "https://github.com/login/oauth/access_token",
      {
        client_id: GITHUB_CLIENT_ID,
        client_secret: GITHUB_CLIENT_SECRET,
        code,
        redirect_uri: redirectUri,
      },
      { headers: { Accept: "application/json" } }
    );

    const accessToken = tokenRes.data.access_token;

    if (!accessToken) {
      console.error("Token exchange failed:", tokenRes.data);
      return res.status(500).send("OAuth failed.");
    }

    // Save to session
    console.log("Token acquired. Handing off to frontend...");

    return res.redirect(`${FRONTEND_URL}?token=${accessToken}`);
  } catch (err) {
    console.error("OAuth callback error:", err);
    return res.status(500).send("OAuth failed.");
  }
});

/* 
   STEP 3 — BACKEND BRIDGE REDIRECT
 */
router.get("/redirect", (req, res) => {
  // Cookie was set during callback, safe to redirect to frontend now
  console.log("DEBUG REDIRECT: Cookie should now be stored.");
  res.redirect(FRONTEND_URL);
});

/* STEP 4 — CHECK AUTH STATE */
router.get("/me", async (req, res) => {
  const authHeader = req.headers.authorization;
  const token = authHeader ? authHeader.split(" ")[1] : null;

  if (!token) {
    return res.status(401).json({ error: "Not authenticated" });
  }

  try {
    const octokit = new Octokit({ auth: token });
    const { data: user } = await octokit.rest.users.getAuthenticated();

    res.json(user);
  } catch (err) {
    console.error("User fetch error:", err);
    return res.status(401).json({ error: "Invalid token" });
  }
});

/* 
   STEP 5 — LOGOUT
 */
router.post("/logout", (_req, res) => {
  // Since we use Stateless Auth (Tokens), there is no server session to destroy.
  // The frontend handles logout by removing the token from LocalStorage.

  res.json({ message: "Logged out successfully" });
});

export default router;
