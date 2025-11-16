// ============================================================================
// hilbert_pipeline_api.js - Hilbert Pipeline Events & Orchestration Helpers
// ============================================================================
//
// This module complements hilbert_api.js by handling:
//   - fetching the pipeline step plan
//   - connecting to the /ws/pipeline WebSocket
//   - structured event handling for step_start / step_complete / step_error / log / artifact
// ============================================================================

const API_BASE = "/api/v1";

// ---------------------------------------------------------------------------
// Fetch pipeline plan
// ---------------------------------------------------------------------------
export async function getPipelinePlan() {
  const res = await fetch(`${API_BASE}/get_pipeline_plan`);
  if (!res.ok) {
    throw new Error(`Failed to fetch pipeline plan: ${res.status}`);
  }
  const data = await res.json();
  return data.steps || [];
}

// ---------------------------------------------------------------------------
// Connect to pipeline WebSocket stream
// ---------------------------------------------------------------------------
//
// onEvent(evt) will receive:
// {
//   type: "step_start" | "step_complete" | "step_error" | "log" | "artifact",
//   timestamp: number,
//   step: string | null,
//   data: {...}
// }
//
// Returns a function you can call to close the connection.
// ---------------------------------------------------------------------------
export function connectPipelineStream(onEvent) {
  const loc = window.location;
  const proto = loc.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${proto}://${loc.host}/ws/pipeline`;

  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    // Optional: send a hello if you want
    // ws.send(JSON.stringify({ type: "hello", source: "pipeline-panel" }));
  };

  ws.onmessage = (msg) => {
    try {
      const evt = JSON.parse(msg.data);
      if (onEvent) onEvent(evt);
    } catch (err) {
      console.error("[PipelineWS] Failed to parse message:", err);
    }
  };

  ws.onerror = (err) => {
    console.error("[PipelineWS] WebSocket error:", err);
  };

  ws.onclose = () => {
    console.log("[PipelineWS] WebSocket closed");
  };

  // Return a cleanup function
  return () => {
    try {
      ws.close();
    } catch {
      // ignore
    }
  };
}
