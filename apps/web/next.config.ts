import type { NextConfig } from "next";

const lanOrigins = (process.env.HTTPS_SANS || process.env.ALLOWED_DEV_ORIGINS || "")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);

const nextConfig: NextConfig = {
  // Ex. HTTPS_SANS=192.168.1.101 → autorise le hot-reload depuis le téléphone
  allowedDevOrigins: lanOrigins.length ? lanOrigins : undefined,
};

export default nextConfig;
