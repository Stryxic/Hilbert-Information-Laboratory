// ============================================================================
// PipelineOrchestratorPanel.jsx
// ---------------------------------------------------------------------------
// A single, integrated panel that gives you:
//
//   - Run Pipeline button
//   - Live progress per step
//   - Step detail (logs, errors, duration)
//   - Event stream
//   - Artifact browser (JSON, CSV, images, etc.)
//   - Timeline (Gantt-style)
//   - Step inspector (inputs/outputs, files, logs)
//   - Cross-run comparison view
//
// Assumes the following helpers/components already exist:
//   - usePipelineRun        ("./usePipelineRun")
//   - usePipelineStream     ("./usePipelineStream")   [optional live WS]
//   - PipelineProgressPanel ("./PipelineProgressPanel")
//   - PipelineStepViewer    ("./PipelineStepViewer")
//   - PipelineEventStream   ("./PipelineEventStream")
//   - ArtifactBrowser       ("./ArtifactBrowser")
//   - PipelineTimeline      ("./PipelineTimeline")
//   - StepInspector         ("./StepInspector")
//   - PipelineRunCompare    ("./PipelineRunCompare")
// ============================================================================

import React, { useMemo, useState } from "react";

import { usePipelineRun } from "./usePipelineRun";
import { usePipelineStream } from "./usePipelineStream";

import PipelineProgressPanel from "./PipelineProgressPanel";
import PipelineStepViewer from "./PipelineStepViewer";
import PipelineEventStream from "./PipelineEventStream";
import ArtifactBrowser from "./ArtifactBrowser";
import PipelineTimeline from "./PipelineTimeline";
import StepInspector from "./StepInspector";
import PipelineRunCompare from "./PipelineRunCompare";

const palette = {
  bg: "#010409",
  panelBg: "#0d1117",
  border: "#30363d",
  text: "#e6edf3",
  muted: "#8b949e",
  accent: "#58a6ff",
  green: "#10b981",
  red: "#ef4444",
};

// ============================================================================
// MAIN PANEL
// ============================================================================

export default function PipelineOrchestratorPanel({
  // Optional overrides for backend paths
  defaultCorpusDir = "uploaded_corpus",
  defaultOutputDir = "results/hilbert_run",
}) {
  const {
    runState,
    events: httpEvents,
    log,
    errors,
    result,
    runPipeline,
  } = usePipelineRun();

  // Optional WebSocket stream to augment events
  const streamEvents = usePipelineStream();

  // Merge HTTP batch events with streaming ones
  const mergedEvents = useMemo(() => {
    const all = [...httpEvents, ...streamEvents];
    // sort by timestamp if present
    return all
      .filter((e) => e && typeof e === "object")
      .sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));
  }, [httpEvents, streamEvents]);

  const outputDir =
    result?.output_dir || defaultOutputDir || "results/hilbert_run";

  const [selectedStep, setSelectedStep] = useState("");
  const [activeTab, setActiveTab] = useState("step"); // step | artifacts | stream | inspector | timeline | compare

  const stepIds = useMemo(
    () =>
      Array.from(
        new Set(
          mergedEvents
            .filter((e) => e.step)
            .map((e) => e.step)
            .filter(Boolean)
        )
      ),
    [mergedEvents]
  );

  const handleRun = () => {
    runPipeline(defaultCorpusDir, defaultOutputDir);
  };

  // ---------------------------------------------------------------------------
  // RENDER
  // ---------------------------------------------------------------------------

  return (
    <div
      className="h-full w-full flex flex-col"
      style={{ background: palette.bg, color: palette.text }}
    >
      {/* HEADER BAR */}
      <div
        className="flex items-center justify-between px-4 py-2 border-b"
        style={{ borderColor: palette.border, background: "#020817" }}
      >
        <div className="flex flex-col">
          <span className="text-sm font-semibold">
            Hilbert Pipeline Orchestrator
          </span>
          <span className="text-xs" style={{ color: palette.muted }}>
            Run, inspect, and compare Hilbert Information Chemistry pipeline
            executions.
          </span>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div>
            Status:{" "}
            <span
              style={{
                color:
                  runState === "running"
                    ? palette.accent
                    : runState === "error"
                    ? palette.red
                    : palette.green,
              }}
            >
              {runState}
            </span>
          </div>

          <button
            className="px-3 py-1 rounded-lg text-sm font-semibold"
            style={{
              background: "#238636",
              color: "#ffffff",
              border: `1px solid ${palette.border}`,
            }}
            onClick={handleRun}
            disabled={runState === "running"}
          >
            {runState === "running" ? "Running…" : "Run Pipeline"}
          </button>
        </div>
      </div>

      {/* BODY GRID */}
      <div className="flex-1 grid grid-cols-12 gap-3 p-3 overflow-hidden">
        {/* LEFT COLUMN: PROGRESS + ERRORS */}
        <div className="col-span-3 flex flex-col gap-3 min-h-0">
          <div className="flex-1 min-h-0">
            <PipelineProgressPanel events={mergedEvents} />
          </div>

          {errors && errors.length > 0 && (
            <div
              className="p-2 rounded-xl border text-xs"
              style={{
                borderColor: palette.border,
                background: palette.panelBg,
                color: palette.red,
              }}
            >
              <div className="font-semibold mb-1">Errors</div>
              {errors.map((e, i) => (
                <div key={i}>{e}</div>
              ))}
            </div>
          )}

          <div className="text-[10px] text-[#8b949e] mt-auto">
            Output directory: {outputDir}
          </div>
        </div>

        {/* CENTER COLUMN: TABS (Step / Artifacts / Stream / Inspector / Timeline / Compare) */}
        <div className="col-span-9 flex flex-col gap-3 min-h-0">
          {/* TAB STRIP */}
          <div
            className="flex items-center gap-2 px-3 py-2 rounded-xl border"
            style={{ borderColor: palette.border, background: palette.panelBg }}
          >
            {[
              { id: "step", label: "Step Detail" },
              { id: "artifacts", label: "Artifacts" },
              { id: "stream", label: "Event Stream" },
              { id: "inspector", label: "Step Inspector" },
              { id: "timeline", label: "Timeline" },
              { id: "compare", label: "Compare Runs" },
            ].map((tab) => {
              const selected = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className="px-2 py-1 rounded text-xs"
                  style={{
                    border: `1px solid ${
                      selected ? palette.accent : palette.border
                    }`,
                    background: selected ? "#161b22" : "transparent",
                    color: selected ? palette.accent : palette.muted,
                  }}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* TAB CONTENT GRID */}
          <div className="flex-1 grid grid-cols-12 gap-3 min-h-0">
            {/* LEFT: STEP SELECTION (for step-detail tab) */}
            {activeTab === "step" && (
              <>
                <div className="col-span-3 flex flex-col min-h-0">
                  <div
                    className="mb-2 text-xs"
                    style={{ color: palette.muted }}
                  >
                    Select step
                  </div>
                  <select
                    className="w-full bg-[#161b22] border border-[#30363d] rounded p-1 text-xs"
                    value={selectedStep}
                    onChange={(e) => setSelectedStep(e.target.value)}
                  >
                    <option value="">Choose…</option>
                    {stepIds.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>

                  <div className="mt-3 text-[11px]" style={{ color: palette.muted }}>
                    Steps are populated from live pipeline events. If you do not
                    see any, run the pipeline or load a previous run.
                  </div>
                </div>

                <div className="col-span-9 min-h-0">
                  {selectedStep ? (
                    <PipelineStepViewer
                      events={mergedEvents}
                      stepId={selectedStep}
                    />
                  ) : (
                    <div
                      className="h-full flex items-center justify-center rounded-xl border text-xs"
                      style={{
                        borderColor: palette.border,
                        background: palette.panelBg,
                        color: palette.muted,
                      }}
                    >
                      Select a step to view logs and details.
                    </div>
                  )}
                </div>
              </>
            )}

            {/* ARTIFACTS TAB */}
            {activeTab === "artifacts" && (
              <div className="col-span-12 min-h-0">
                <ArtifactBrowser outputDir={outputDir} />
              </div>
            )}

            {/* EVENT STREAM TAB */}
            {activeTab === "stream" && (
              <div className="col-span-12 min-h-0">
                <PipelineEventStream events={mergedEvents} />
              </div>
            )}

            {/* STEP INSPECTOR TAB */}
            {activeTab === "inspector" && (
              <div className="col-span-12 min-h-0">
                <StepInspector events={mergedEvents} outputDir={outputDir} />
              </div>
            )}

            {/* TIMELINE TAB */}
            {activeTab === "timeline" && (
              <div className="col-span-12 min-h-0">
                <PipelineTimeline events={mergedEvents} />
              </div>
            )}

            {/* COMPARE TAB */}
            {activeTab === "compare" && (
              <div className="col-span-12 min-h-0">
                <PipelineRunCompare />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* RAW PIPELINE LOG (optional footer) */}
      <div
        className="px-3 py-2 border-t text-[11px] overflow-x-auto"
        style={{ borderColor: palette.border, background: "#02040c" }}
      >
        <span className="font-semibold mr-2">Log:</span>
        <span style={{ color: palette.muted }}>
          {log && log.length ? log.join(" | ") : "No messages yet."}
        </span>
      </div>
    </div>
  );
}
