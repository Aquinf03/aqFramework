import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  async redirects() {
    return [
      { source: "/login", destination: "/", permanent: false },
      {
        source: "/auth/desktop",
        destination: "/?view=desktop",
        permanent: false,
      },
      {
        source: "/app",
        destination: "/",
        permanent: false,
      },
      {
        source: "/app/:path*",
        destination: "/",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
