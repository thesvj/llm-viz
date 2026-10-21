/** @type {import('next').NextConfig} */

let withBundleAnalyzer = require("@next/bundle-analyzer")({
    enabled: process.env.ANALYZE === "true",
});

const nextConfig = {
  reactStrictMode: false,
  productionBrowserSourceMaps: true,
  // Always static HTML export for Cloudflare Pages / any static host.
  // Dev-only file-write tooling lives outside the App Router (see scripts/dev-schematic-api.mjs).
  output: 'export',
  // GitHub Pages serves this repo at /llm-viz. Set NEXT_PUBLIC_BASE_PATH=/llm-viz for that build.
  // Pages serves a folder's index.html, so those builds also need a trailing slash.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || undefined,
  trailingSlash: Boolean(process.env.NEXT_PUBLIC_BASE_PATH),
};

module.exports = withBundleAnalyzer(nextConfig);
