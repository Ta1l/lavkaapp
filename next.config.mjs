/** @type {import('next').NextConfig} */
const config = {
  transpilePackages: ['date-fns'],
  experimental: {
    esmExternals: 'loose'
  }
};

export default config; 