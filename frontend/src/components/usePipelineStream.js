// ============================================================================
// usePipelineStream.js — Real-time pipeline event stream via WebSocket
// ============================================================================

import { useState, useEffect } from "react";

export function usePipelineStream() {
  const [events, setEvents] = useState([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/pipeline");

    ws.onmessage = (msg) => {
      try {
        const evt = JSON.parse(msg.data);
        setEvents((prev) => [...prev, evt]);
      } catch {}
    };

    return () => ws.close();
  }, []);

  return events;
}
