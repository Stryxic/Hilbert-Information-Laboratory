// ============================================================================
// hilbert_plugins.jsx
// ----------------------------------------------------------------------------
// Declares all available dashboard panels for the Hilbert Lab UI.
//
// Each plugin:
//   - has a unique `id`
//   - human-readable `title`
//   - `component`: a React component TYPE (not JSX)
// ============================================================================

import OutputPanel from "./OutputPanel";  // <-- NEW unified panel

import DocumentsPanel from "./DocumentsPanel";
import PeriodicTablePanel from "./PeriodicTablePanel";
import GraphSelectorPanel from "./GraphSelectorPanel";
import MoleculeViewer from "./MoleculeViewer";
import PersistencePanel from "./PersistencePanel";
import ReportPanel from "./ReportPanel";
// import DraftEditorPanel from "./DraftEditorPanel";  // optional


export const PLUGINS = [
  // ==========================================================================
  // NEW — Unified Hilbert Output Multiplexer Panel
  // ==========================================================================
  {
    id: "output-panel",
    title: "Hilbert Output Suite",
    component: OutputPanel,
  },

  // ==========================================================================
  // Existing panels
  // ==========================================================================
  {
    id: "documents-panel",
    title: "Documents",
    component: DocumentsPanel,
  },
  {
    id: "periodic-table",
    title: "Informational Periodic Table",
    component: PeriodicTablePanel,
  },
  {
    id: "graph-selector",
    title: "Graph & Molecule Selector",
    component: GraphSelectorPanel,
  },
  {
    id: "molecule-viewer",
    title: "Molecular Stability Map",
    component: MoleculeViewer,
  },
  {
    id: "persistence-panel",
    title: "Persistence & Stability Field",
    component: PersistencePanel,
  },
  {
    id: "report-panel",
    title: "Reports & Exports",
    component: ReportPanel,
  },

  // Optional:
  // {
  //   id: "draft-editor",
  //   title: "Draft Editor",
  //   component: DraftEditorPanel,
  // },
];
