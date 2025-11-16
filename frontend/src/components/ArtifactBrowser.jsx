// ============================================================================
// ArtifactBrowser.jsx — Dynamic viewer for any pipeline artifact
// ============================================================================

import React, { useState, useEffect } from "react";

export default function ArtifactBrowser({ outputDir }) {
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [content, setContent] = useState(null);

  // Load artifact list
  useEffect(() => {
    fetch(`/api/v1/list_artifacts?dir=${outputDir}`)
      .then((r) => r.json())
      .then((d) => setFiles(d.files || []));
  }, [outputDir]);

  // Load content when activeFile changes
  useEffect(() => {
    if (!activeFile) return;

    fetch(`/api/v1/get_artifact?path=${encodeURIComponent(activeFile)}`)
      .then((r) => r.json())
      .then((d) => setContent(d));
  }, [activeFile]);

  return (
    <div className="h-full grid grid-cols-12 gap-3">
      {/* LEFT: FILE LIST */}
      <div className="col-span-3 bg-[#0d1117] border border-[#30363d] rounded-xl p-2 overflow-y-auto text-xs">
        <div className="font-semibold text-[#8b949e] mb-2">Artifacts</div>
        {files.map((f) => (
          <div
            key={f}
            onClick={() => setActiveFile(f)}
            className="cursor-pointer hover:bg-[#161b22] p-1 rounded"
          >
            {f}
          </div>
        ))}
      </div>

      {/* RIGHT: FILE PREVIEW */}
      <div className="col-span-9 bg-[#0d1117] border border-[#30363d] rounded-xl p-3 overflow-auto text-xs">
        {!content && (
          <div className="text-[#8b949e]">Select an artifact to view.</div>
        )}

        {/* JSON */}
        {content?.type === "json" && (
          <pre className="whitespace-pre-wrap">
            {JSON.stringify(content.data, null, 2)}
          </pre>
        )}

        {/* CSV → table */}
        {content?.type === "csv" && (
          <table className="table-auto text-xs">
            <thead>
              <tr>
                {content.columns.map((c) => (
                  <th key={c} className="px-2 py-1 border border-[#30363d]">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {content.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td
                      key={j}
                      className="px-2 py-1 border border-[#30363d]"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* PNG / JPG */}
        {content?.type === "image" && (
          <img
            src={`/api/v1/get_artifact_raw?path=${encodeURIComponent(activeFile)}`}
            alt=""
            className="max-w-full"
          />
        )}

        {/* TXT */}
        {content?.type === "text" && (
          <pre className="whitespace-pre-wrap">
            {content.data}
          </pre>
        )}

        {/* PDF */}
        {content?.type === "pdf" && (
          <iframe
            className="w-full h-full"
            src={`/api/v1/get_artifact_raw?path=${encodeURIComponent(activeFile)}`}
            title="PDF"
          />
        )}
      </div>
    </div>
  );
}
