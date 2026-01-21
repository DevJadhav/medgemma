/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',  // Required for Docker deployment
  env: {
    API_URL: process.env.API_URL || 'http://api:8000',
  },
  // Increase timeout for AI inference requests
  experimental: {
    proxyTimeout: 300000, // 5 minutes for long AI inference
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_URL || 'http://api:8000'}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
