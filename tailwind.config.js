/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Jura', 'system-ui', 'sans-serif'],
        jura: ['Jura', 'system-ui', 'sans-serif']
      },
      fontMetrics: {
        jura: {
          capHeight: 700,
          ascent: 1000,
          descent: -200,
          lineGap: 0,
          unitsPerEm: 1000,
          xHeight: 500,
        },
      },
    },
  },
  plugins: [
    function({ addBase, theme }) {
      addBase({
        'body': {
          fontFamily: theme('fontFamily.sans'),
          fontSizeAdjust: '0.5',
          '@media screen and (prefers-reduced-motion: no-preference)': {
            scrollBehavior: 'smooth',
          },
        },
      });
    },
  ],
} 