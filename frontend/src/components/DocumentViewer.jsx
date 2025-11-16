// ============================================================================
// DocumentViewer.jsx — Full-token version with Hilbert-vs-Document color coding
// Hilbert Information Chemistry Lab — Production Version
// ============================================================================

import React, {
  useEffect,
  useMemo,
  useState,
  useCallback,
} from "react";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000/api/v1";

function asNumber(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}

function pickWeight(row) {
  const fields = [
    "entropy",
    "coherence",
    "tf",
    "df",
    "info_score",
    "misinfo_score",
    "disinfo_score",
  ];
  for (const f of fields) {
    if (row && f in row) {
      const v = asNumber(row[f]);
      if (v !== 0) return v;
    }
  }
  return 0;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================
export default function DocumentViewer({
  docName,
  elementsForDoc = [],
  compounds = [],
  elementClusters = {},
  onSelectElement,
  onClose,
  appendLog,
}) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [sortMode, setSortMode] = useState("freq");
  const [sortDir, setSortDir] = useState("desc");

  const [selectedEl, setSelectedEl] = useState(null);

  // ---------------------------------------------------------------------------
  // Load document raw text
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!docName) return;

    (async () => {
      setLoading(true);
      setError("");
      setSelectedEl(null);

      try {
        const res = await axios.get(`${API_BASE}/get_document_text`, {
          params: { name: docName },
        });
        setText(res.data?.text || "");
      } catch (err) {
        console.error("[DocumentViewer] load failed:", err);
        setError("Failed to load document text.");
      } finally {
        setLoading(false);
      }
    })();
  }, [docName]);

  // ---------------------------------------------------------------------------
  // Build master token list: raw tokens + hilbert tokens merged
  // ---------------------------------------------------------------------------
  const mergedTokens = useMemo(() => {
    if (!text) return [];

    // 1. Tokenize the document text (simple word tokenizer)
    const words =
      text.toLowerCase().match(/[a-z][a-z'\-]+/gi) || [];

    const freq = {};
    for (const w of words) {
      freq[w] = (freq[w] || 0) + 1;
    }

    // 2. Hilbert tokens present in this document
    const hilbertTokens = new Set(
      elementsForDoc.map((r) =>
        (r.token || r.element || "")
          .toLowerCase()
          .trim()
      )
    );

    // 3. Build combined record for each token
    const records = Object.entries(freq).map(([token, count]) => {
      const isHilbert = hilbertTokens.has(token);

      // Metadata from Hilbert if available
      let row = null;
      if (isHilbert) {
        row = elementsForDoc.find(
          (r) =>
            (r.token || r.element || "")
              .toLowerCase()
              === token
        );
      }

      const weight = row ? pickWeight(row) : 0;

      const code =
        (row &&
          (row.element ||
            row.token ||
            row.id ||
            row.code)) ||
        token;

      const root =
        (row &&
          (elementClusters[row.element] ||
            elementClusters[token] ||
            row.element)) ||
        token;

      return {
        code: token,
        label: token,
        freq: count,
        weight,
        root,
        isHilbert,
      };
    });

    return records.sort((a, b) => b.freq - a.freq);
  }, [text, elementsForDoc, elementClusters]);

  // ---------------------------------------------------------------------------
  // Sorting logic
  // ---------------------------------------------------------------------------
  const sortedTokens = useMemo(() => {
    const arr = [...mergedTokens];
    const dir = sortDir === "asc" ? 1 : -1;

    if (sortMode === "freq") {
      arr.sort((a, b) => (b.freq - a.freq) * dir);
    } else if (sortMode === "weight") {
      arr.sort(
        (a, b) =>
          (b.weight - a.weight) * dir ||
          b.freq - a.freq
      );
    } else if (sortMode === "alpha") {
      arr.sort((a, b) =>
        a.label.localeCompare(b.label)
      );
    } else if (sortMode === "root") {
      arr.sort((a, b) => {
        if (a.root === b.root)
          return b.freq - a.freq;
        return a.root.localeCompare(b.root);
      });
    }

    return arr;
  }, [mergedTokens, sortMode, sortDir]);

  // ---------------------------------------------------------------------------
  // Highlighting in the text
  // ---------------------------------------------------------------------------
  const highlightedHTML = useMemo(() => {
    if (!text) return "";
    if (mergedTokens.length === 0)
      return text.replace(/\n/g, "<br>");

    const tokens = mergedTokens
      .map((t) => t.label)
      .sort((a, b) => b.length - a.length);

    const escaped = tokens.map((t) =>
      escapeRegex(t)
    );
    const pattern = `(?<![\\w-])(${escaped.join(
      "|"
    )})(?![\\w-])`;
    const re = new RegExp(pattern, "gi");

    let count = 0;
    let html = text.replace(re, (m) => {
      const key = m.toLowerCase();
      count++;

      const tokenMeta = mergedTokens.find(
        (t) => t.code === key
      );

      const cls = tokenMeta?.isHilbert
        ? "hilbert-tok"
        : "doc-tok";

      return `<span class="${cls}" data-el="${key}">${m}</span>`;
    });

    html = html.replace(/\n/g, "<br>");
    console.log(
      `[highlight] ${count} tokens highlighted for ${docName}`
    );
    return html;
  }, [text, mergedTokens, docName]);

  // ---------------------------------------------------------------------------
  // Element selection
  // ---------------------------------------------------------------------------
  const handleElementSelect = useCallback(
    (code) => {
      const item =
        sortedTokens.find((e) => e.code === code) ||
        mergedTokens.find((e) => e.code === code) ||
        { code, label: code };

      setSelectedEl(item);

      onSelectElement?.({
        element: item.code,
        label: item.label,
        doc: docName,
        source: "document-viewer",
      });

      appendLog?.(
        `[DocumentViewer] Selected token '${item.code}'`
      );
    },
    [
      sortedTokens,
      mergedTokens,
      onSelectElement,
      appendLog,
      docName,
    ]
  );

  const handleTextClick = (e) => {
    const el = e.target?.dataset?.el;
    if (el) handleElementSelect(el);
  };

  // ---------------------------------------------------------------------------
  // Compound correlation
  // ---------------------------------------------------------------------------
  const relatedCompounds = useMemo(() => {
    if (!selectedEl) return [];

    const key = selectedEl.code;

    return (compounds || []).filter((c) => {
      let els = c.elements || c.els || [];
      if (typeof els === "string") {
        els = els
          .split(/[,\s]+/)
          .map((s) => s.trim());
      }
      return Array.isArray(els) && els.includes(key);
    });
  }, [selectedEl, compounds]);

  // ========================================================================
  // RENDER
  // ========================================================================
  return (
    <div
      style={{
        display: "flex",
        height: "100%",
        background: "#0d1117",
        border: "1px solid #30363d",
        borderRadius: 12,
        overflow: "hidden",
      }}
    >
      {/* LEFT: DOCUMENT TEXT */}
      <div
        style={{
          flex: 2,
          display: "flex",
          flexDirection: "column",
          borderRight: "1px solid #30363d",
        }}
      >
        <div
          style={{
            padding: "8px 10px",
            background: "#161b22",
            borderBottom: "1px solid #30363d",
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span style={{ fontWeight: 600, fontSize: 13 }}>
            {docName}
          </span>
          <button
            style={{
              border: "none",
              background: "none",
              color: "#8b949e",
              cursor: "pointer",
            }}
            onClick={onClose}
          >
            ✕
          </button>
        </div>

        <div
          onClick={handleTextClick}
          style={{
            flex: 1,
            overflowY: "auto",
            padding: 16,
            color: "#e6edf3",
            fontSize: 13,
            lineHeight: 1.6,
          }}
          dangerouslySetInnerHTML={{
            __html: loading
              ? "<em>Loading…</em>"
              : error
              ? `<span style='color:#ef4444'>${error}</span>`
              : highlightedHTML,
          }}
        />

        <style>
          {`
            .hilbert-tok {
              background: rgba(88,166,255,0.25);
              border-bottom: 1px solid #58a6ff;
              cursor: pointer;
              transition: background 0.15s;
            }
            .hilbert-tok:hover {
              background: rgba(88,166,255,0.40);
            }

            .doc-tok {
              background: rgba(139,148,158,0.20);
              border-bottom: 1px dotted #8b949e;
              cursor: pointer;
              transition: background 0.15s;
            }
            .doc-tok:hover {
              background: rgba(139,148,158,0.35);
            }
          `}
        </style>
      </div>

      {/* RIGHT: SIDEBAR */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: "8px 10px",
          background: "#151b23",
        }}
      >
        {/* Sorting controls */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            borderBottom: "1px solid #202938",
            paddingBottom: 4,
          }}
        >
          <span style={{ fontSize: 12, fontWeight: 600 }}>
            Elements in this document
          </span>

          <div
            style={{
              display: "flex",
              gap: 4,
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: 10, color: "#8b949e" }}>
              Sort:
            </span>
            <select
              value={sortMode}
              onChange={(e) => {
                const v = e.target.value;
                setSortMode(v);
                if (v === "alpha" || v === "root")
                  setSortDir("asc");
                else if (sortDir === "asc")
                  setSortDir("desc");
              }}
              style={{
                background: "#0d1117",
                color: "#e6edf3",
                border: "1px solid #30363d",
                borderRadius: 6,
                padding: "2px 4px",
                fontSize: 10,
              }}
            >
              <option value="freq">Frequency</option>
              <option value="weight">Weight</option>
              <option value="alpha">A–Z</option>
              <option value="root">Root</option>
            </select>

            {(sortMode === "freq" ||
              sortMode === "weight") && (
              <button
                onClick={() =>
                  setSortDir((d) =>
                    d === "desc" ? "asc" : "desc"
                  )
                }
                style={{
                  padding: "0 4px",
                  background: "none",
                  border: "1px solid #30363d",
                  borderRadius: 4,
                  color: "#8b949e",
                  cursor: "pointer",
                }}
              >
                {sortDir === "desc" ? "↓" : "↑"}
              </button>
            )}
          </div>
        </div>

        {/* Token chips */}
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            paddingTop: 6,
            display: "flex",
            flexWrap: "wrap",
            gap: 6,
          }}
        >
          {sortedTokens.map((el) => {
            const isSel = selectedEl?.code === el.code;

            const bg = el.isHilbert
              ? "rgba(88,166,255,0.20)"
              : "rgba(139,148,158,0.15)";

            const border = el.isHilbert
              ? (isSel ? "#58a6ff" : "#30363d")
              : (isSel ? "#8b949e" : "#30363d");

            const color = el.isHilbert
              ? (isSel ? "#58a6ff" : "#e6edf3")
              : (isSel ? "#c9d1d9" : "#9da7b3");

            return (
              <div
                key={el.code}
                onClick={() => handleElementSelect(el.code)}
                style={{
                  padding: "3px 6px",
                  borderRadius: 999,
                  background: bg,
                  border: `1px solid ${border}`,
                  color,
                  cursor: "pointer",
                  fontSize: 10,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                }}
                title={`${el.isHilbert ? "Hilbert element" : "Document-only token"}
freq=${el.freq}${el.weight ? `, weight=${el.weight.toFixed(2)}` : ""}`}
              >
                <span>{el.label}</span>
                <span style={{ fontSize: 8 }}>{el.freq}</span>

                {el.weight > 0 && (
                  <span
                    style={{
                      fontSize: 8,
                      color: "#10b981",
                    }}
                  >
                    {el.weight.toFixed(2)}
                  </span>
                )}

                {el.root && el.root !== el.code && (
                  <span
                    style={{
                      fontSize: 8,
                      color: "#facc15",
                    }}
                  >
                    {el.root}
                  </span>
                )}

                {!el.isHilbert && (
                  <span
                    style={{
                      fontSize: 7,
                      background: "#30363d",
                      borderRadius: 4,
                      padding: "0 3px",
                      color: "#9da7b3",
                    }}
                  >
                    doc
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Compounds */}
        <div
          style={{
            borderTop: "1px solid #202938",
            paddingTop: 6,
            fontSize: 10,
            maxHeight: 140,
            overflowY: "auto",
          }}
        >
          {!selectedEl && (
            <div style={{ color: "#8b949e" }}>
              Click a term to inspect its details.
            </div>
          )}

          {selectedEl && relatedCompounds.length === 0 && (
            <div style={{ color: "#8b949e" }}>
              <strong>{selectedEl.code}</strong> is not in
              any compound.
            </div>
          )}

          {selectedEl && relatedCompounds.length > 0 && (
            <div>
              <strong>{selectedEl.code}</strong> appears in:
              {relatedCompounds.map((c, i) => (
                <div key={i} style={{ marginTop: 4 }}>
                  <span style={{ color: "#58a6ff" }}>
                    {c.compound_id || c.id || `Compound ${i}`}
                  </span>
                  {c.stability && (
                    <span
                      style={{
                        marginLeft: 4,
                        color: "#10b981",
                      }}
                    >
                      S={asNumber(c.stability).toFixed(2)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
