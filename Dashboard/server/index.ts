import express from "express";
import cors from "cors";

import authRouter from "./auth.js";
import { registerRoutes } from "./routes.js";

const app = express();

/* IMPORTANT: TRUST PROXY (required for Railway HTTPS)
 */
app.set("trust proxy", 1);

/*
   FRONTEND URL
   (Reads from Railway Variables or defaults to localhost)
 */
const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:3000";

/* CORS
 */
app.use(
  cors({
    origin: FRONTEND_URL,
    credentials: true,
  })
);

/* BODY PARSERS
 */
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

/* ROUTES
 */
app.use("/api/auth", authRouter);

(async () => {
  await registerRoutes(app);

  app.get("/", (_req, res) => {
    res.json({ status: "Backend running on Railway (Stateless)" });
  });

  const port = parseInt(process.env.PORT || "5000", 10);
  app.listen(port, "0.0.0.0", () => {
    console.log(`Backend running on port ${port}`);
  });
})();
