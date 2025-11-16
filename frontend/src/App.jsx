// src/App.jsx
import React from "react";
import Dashboard from "./components/Dashboard";
import "./styles/hilbert_lab.css";

/**
 * Hilbert Information Chemistry Lab — App Root
 * -------------------------------------------------------
 * This version removes old tab navigation (Dashboard / Periodic Table)
 * and renders the unified Dashboard interface directly.
 * The header bar and controls are now part of Dashboard itself.
 */
export default function App() {
  return (
    <div className="app-root">
      <Dashboard />
    </div>
  );
}
