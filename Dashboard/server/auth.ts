// // Extend Express Request type to include session
// declare module "express-session" {
//   interface SessionData {
//     accessToken?: string;
//   }
// }
// import express, { Request, Response } from "express";
// import axios from "axios";
// import session from "express-session";
// import { Octokit } from "@octokit/rest";

// const router = express.Router();

// // Hardcoded GitHub OAuth credentials (safe to expose client_id)
// const CLIENT_ID = "Ov23liYe2zHfm9WetpmF"; // <-- Replace with your actual client_id
// const CLIENT_SECRET = "c4172408ee26e8a3f8744e03c21578636e728fea"; // <-- NEVER expose to frontend
// const REDIRECT_URI = "http://localhost:3000/api/auth/github/callback";

// router.use(session({
//   secret: "supersecretkey",
//   resave: false,
//   saveUninitialized: true,
//   cookie: { secure: false }, // Set to true if using HTTPS
// }));

// router.get("/github", (req: Request, res: Response) => {
//   const githubAuthUrl = `https://github.com/login/oauth/authorize?client_id=${CLIENT_ID}&redirect_uri=${encodeURIComponent(REDIRECT_URI)}`;
//   res.redirect(githubAuthUrl);
// });

// router.get("/github/callback", async (req: Request, res: Response) => {
//   const code = req.query.code as string;
//   if (!code) return res.status(400).send("No code provided");

//   try {
//     const tokenRes = await axios.post(
//       "https://github.com/login/oauth/access_token",
//       {
//         client_id: CLIENT_ID,
//         client_secret: CLIENT_SECRET,
//         code,
//         redirect_uri: REDIRECT_URI,
//       },
//       {
//         headers: { Accept: "application/json" },
//       }
//     );
//     const accessToken = tokenRes.data.access_token;
//     if (!accessToken) return res.status(401).send("No access token");
//     req.session.accessToken = accessToken;
//     res.redirect("/dashboard");
//   } catch (err) {
//     res.status(500).send("GitHub OAuth failed");
//   }
// });

// router.get("/me", async (req: Request, res: Response) => {
//   const accessToken = req.session.accessToken;
//   if (!accessToken) return res.status(401).json({ error: "Not authenticated" });
//   try {
//     const octokit = new Octokit({ auth: accessToken });
//     const { data: user } = await octokit.rest.users.getAuthenticated();
//     res.json(user);
//   } catch (err) {
//     res.status(500).json({ error: "Failed to fetch user info" });
//   }
// });

// export default router;


// SWE_project_website\server\auth.ts
import dotenv from "dotenv";
dotenv.config();  // MUST be first

import express, { Request, Response } from "express";
import axios from "axios";
import { Octokit } from "@octokit/rest";
import session from "express-session";

declare module "express-session" {
  interface SessionData {
    accessToken?: string;
  }
}

const router = express.Router();

console.log("AUTH ENV TEST (auth.ts):", process.env.GITHUB_CLIENT_ID, process.env.GITHUB_REDIRECT_URI);

// Redirect user to GitHub OAuth
router.get("/github", (_req: Request, res: Response) => {
  const CLIENT_ID = process.env.GITHUB_CLIENT_ID;
  const REDIRECT_URI = process.env.GITHUB_REDIRECT_URI;

  // 💡 NEW: Guard clause to check for missing ENV variables
  if (!CLIENT_ID || !REDIRECT_URI) {
    console.error("CRITICAL: GITHUB_CLIENT_ID or GITHUB_REDIRECT_URI is missing.");
    // Respond with a 500 error if configuration is missing
    return res.status(500).send("Server configuration error: Missing GitHub credentials.");
  }
  
  console.log("AUTH /github → ENV:", CLIENT_ID, REDIRECT_URI);

  const url =
    `https://github.com/login/oauth/authorize` +
    `?client_id=${CLIENT_ID}&redirect_uri=${encodeURIComponent(REDIRECT_URI)}` +
    `&scope=repo,user`; // Added necessary scopes for PR review

  res.redirect(url);
});

// GitHub redirects here with "code"
router.get("/github/callback", async (req: Request, res: Response) => {
  const CLIENT_ID = process.env.GITHUB_CLIENT_ID;
  const CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET;
  const REDIRECT_URI = process.env.GITHUB_REDIRECT_URI;
  const FRONTEND_URL = process.env.FRONTEND_URL;

  if (!CLIENT_ID || !CLIENT_SECRET || !REDIRECT_URI || !FRONTEND_URL) {
    console.error("CRITICAL: One or more OAuth ENV variables are missing in callback.");
    return res.status(500).send("Server configuration error: Incomplete OAuth setup.");
  }
  
  console.log("AUTH CALLBACK ENV:", CLIENT_ID, REDIRECT_URI, FRONTEND_URL);

  const code = req.query.code as string;
  if (!code) {
    // Check if GitHub returned an error instead of a code
    const githubError = req.query.error;
    if (githubError) {
      console.error(`GitHub Auth Error: ${githubError}`);
      return res.status(400).send(`GitHub Auth Error: ${githubError}`);
    }
    return res.status(400).send("Code missing");
  }

  try {
    const tokenRes = await axios.post(
      "https://github.com/login/oauth/access_token",
      {
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
        code,
        redirect_uri: REDIRECT_URI,
      },
      { headers: { Accept: "application/json" } }
    );

    const accessToken = tokenRes.data.access_token;
    if (!accessToken) {
      console.error("GitHub access token exchange failed.", tokenRes.data);
      return res.status(401).send("No access token or token exchange failed.");
    }

    req.session.accessToken = accessToken;

    // 💡 Refinement: Ensure we redirect to the specific frontend URL, not just the path
    res.redirect(FRONTEND_URL); 
  } catch (err) {
    console.error("Auth failed during token exchange:", err);
    res.status(500).send("Auth failed");
  }
});

// Get logged-in user info
router.get("/me", async (req: Request, res: Response) => {
  if (!req.session.accessToken)
    return res.status(401).json({ error: "Not authenticated" });

  try {
    const octokit = new Octokit({ auth: req.session.accessToken });
    const { data: user } = await octokit.rest.users.getAuthenticated();
    res.json(user);
  } catch (err) {
    console.error("Octokit failed to get user info:", err);
    // Clearing session if Octokit fails, forcing re-login
    delete req.session.accessToken; 
    return res.status(401).json({ error: "Token invalid, please log in again." });
  }
});

export default router;