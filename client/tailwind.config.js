/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#060A10",
        panel: "#0C1018",
        panel2: "#0A0E16",
        border: "rgba(148,163,184,0.08)",
        text: "#E2E8F0",
        muted: "rgba(148,163,184,0.55)",
        purple: "#1D4ED8",
        "purple-light": "#E2E8F0",
        purple2: "#1D4ED8",
        accent: "#1D4ED8",
        "accent-light": "#E2E8F0",
        danger: "#DC4A4A",
        success: "#3AAB5E",
        warn: "#F59E0B",
      },
      fontFamily: {
        sans: ['"Space Grotesk"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "monospace"],
      },
      fontSize: {
        h1: ["48px", { lineHeight: "52px", letterSpacing: "0.2px" }],
        h2: ["32px", { lineHeight: "38px", letterSpacing: "0.2px" }],
        h3: ["20px", { lineHeight: "28px", letterSpacing: "0.2px" }],
        body: ["16px", { lineHeight: "24px" }],
        small: ["13px", { lineHeight: "18px" }],
      },
      borderRadius: {
        DEFAULT: "8px",
        lg: "10px",
        xl: "14px",
      },
    },
  },
  plugins: [],
};
