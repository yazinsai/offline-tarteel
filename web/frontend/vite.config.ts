import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      workbox: {
        globPatterns: ["**/*.{js,css,html,woff,woff2,json}"],
        // Don't precache the ONNX model — it's handled by IndexedDB
        globIgnores: ["**/*.onnx"],
        maximumFileSizeToCacheInBytes: 5 * 1024 * 1024, // 5 MB for quran.json
      },
      manifest: {
        name: "Offline Tarteel",
        short_name: "Tarteel",
        description: "Offline Quran verse recognition in your browser",
        theme_color: "#2c2416",
        background_color: "#faf8f3",
        display: "standalone",
        icons: [
          { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
        ],
      },
    }),
  ],
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
