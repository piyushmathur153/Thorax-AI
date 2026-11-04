/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        medDark: "#070a12",
        medCard: "rgba(19, 29, 51, 0.7)",
        medBorder: "rgba(56, 189, 248, 0.2)",
      }
    },
  },
  plugins: [],
}