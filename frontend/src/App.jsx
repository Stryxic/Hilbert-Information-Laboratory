// src/App.jsx
//
// Minimal Hilbert DB dashboard shell.
//
// - Provides a simple header bar
// - Mounts HilbertDBProvider so children can use the DB / API
// - Renders CorporaPage, which owns the three-pane layout
//

import React from "react";
import { HilbertDBProvider } from "./context/HilbertDBProvider.jsx";
import CorporaPage from "./pages/CorporaPage";

export default function App() {
  return (
    <HilbertDBProvider>
      <div style={styles.appRoot}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.title}>Hilbert Information Laboratory</h1>
            <div style={styles.subtitle}>DB-Integrated Scientific Dashboard</div>
          </div>
        </header>

        <main style={styles.main}>
          <CorporaPage />
        </main>
      </div>
    </HilbertDBProvider>
  );
}

const styles = {
  appRoot: {
    width: "100%",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#f5f5f8",
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    color: "#111827",
  },
  header: {
    flexShrink: 0,
    padding: "16px 24px",
    borderBottom: "1px solid #e5e7eb",
    backgroundColor: "#ffffff",
  },
  title: {
    margin: 0,
    fontSize: "22px",
    fontWeight: 700,
  },
  subtitle: {
    marginTop: 4,
    fontSize: "13px",
    color: "#6b7280",
  },
  main: {
    flex: 1,
    minHeight: 0,
    padding: "16px 24px",
    backgroundColor: "#f5f5f8",
  },
};
