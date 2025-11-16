import React, { useState, useEffect } from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";

export default function DraftEditorPanel({
  activeElement,
  appendLog,
}) {
  const [text, setText] = useState("");
  const [hint, setHint] = useState("");

  useEffect(() => {
    if (!activeElement?.element) {
      setHint("");
      return;
    }
    setHint(
      `You are currently focusing on ${activeElement.element}${
        activeElement.token ? ` (${activeElement.token})` : ""
      }. Draft to reinforce its stable, coherent use.`
    );
  }, [activeElement]);

  const handleSuggest = () => {
    appendLog?.(
      `[Draft] Requesting suggestions for element ${activeElement?.element ||
        "N/A"}`
    );
    // In your real app, you’d hit an assistant endpoint; here we just log.
  };

  return (
    <Card className="h-full flex flex-col bg-[#0d1117] border border-[#30363d] text-xs">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#30363d]">
        <div>
          <div className="font-semibold text-[#e6edf3]">Draft Lab</div>
          <div className="text-[9px] text-[#8b949e]">
            Write, refine, and align text with your informational field.
          </div>
        </div>
        <Button
          size="xs"
          className="bg-[#21262d] hover:bg-[#30363d] text-[9px] px-2 py-1"
          onClick={handleSuggest}
        >
          Get Suggestions
        </Button>
      </div>

      {hint && (
        <div className="px-3 pt-2 text-[8px] text-[#8b949e]">
          {hint}
        </div>
      )}

      <textarea
        className="flex-1 m-3 mt-2 rounded-md bg-[#0d1117] border border-[#30363d] text-[#e6edf3] text-[10px] p-2 outline-none focus:border-[#58a6ff]"
        placeholder="Draft your paragraph or section here. The Hilbert Lab will help keep it stable, coherent, and aligned with your core informational elements."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
    </Card>
  );
}
