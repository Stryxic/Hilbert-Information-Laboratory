// frontend/src/main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import Dashboard from "./components/Dashboard.jsx";
import "./styles/hilbert_lab.css";
import "./index.css";


ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Dashboard />
  </React.StrictMode>
);
