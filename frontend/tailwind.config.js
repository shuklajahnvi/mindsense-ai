/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Nunito", "system-ui", "Segoe UI", "sans-serif"],
        display: ["Fraunces", "Georgia", "serif"],
      },
      colors: {
        // Calm, earthy palette. stone (warm grey) comes from Tailwind.
        sage: {
          50: "#f3f6f1", 100: "#e4ece0", 200: "#cbdac4", 300: "#a9c2a0", 400: "#86a87c",
          500: "#6a8f62", 600: "#557650", 700: "#455f42", 800: "#3a4e38", 900: "#2f3f2e",
        },
        clay: {
          50: "#fbf3ee", 100: "#f5e2d6", 200: "#ecc5ad", 300: "#e0a382", 400: "#d2825d",
          500: "#c4693f", 600: "#a95533", 700: "#8a452c", 800: "#6f3927", 900: "#5b3022",
        },
        ochre: {
          50: "#fcf7ea", 100: "#f7ebc9", 200: "#efd894", 300: "#e6c262", 400: "#dcab3e",
          500: "#c9922b", 600: "#a87522", 700: "#865a20", 800: "#6e4a20", 900: "#5c3e1e",
        },
        sand: {
          50: "#fdfbf7", 100: "#f8f3ea", 200: "#f0e7d8", 300: "#e5d7c0", 400: "#d4c0a0",
          500: "#bfa67f", 600: "#a08660", 700: "#7f6a4c", 800: "#66553e", 900: "#544634",
        },
        dusk: {
          50: "#eef2f9", 100: "#dbe4f2", 200: "#b9c9e4", 300: "#91a9d3", 400: "#6c8cc4",
          500: "#4a74b8", 600: "#3d6199", 700: "#345589", 800: "#2c4670", 900: "#253a5c",
        },
      },
      boxShadow: {
        soft: "0 1px 2px rgba(84,70,52,0.04), 0 10px 30px -12px rgba(84,70,52,0.14)",
        lift: "0 2px 4px rgba(84,70,52,0.05), 0 18px 40px -16px rgba(84,70,52,0.22)",
      },
      borderRadius: { "4xl": "2rem" },
    },
  },
  plugins: [],
};
