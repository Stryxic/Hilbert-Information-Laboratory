// ============================================================================
// ThesisAdvisorPanel.jsx — Element & Compound Narrative Inspector
// ============================================================================
//
// - When activeElement is an element → show semantic profile, regimes, context.
// - When activeElement is a compound → show members, stability & regime summary.
// - Works with selections from PeriodicTablePanel & Molecular Stability Map.
//
// Expected activeElement shapes:
//
//   // From PeriodicTablePanel / others:
//   { type?: "element", element: "E00070", token, label, source }
//
//   // From Molecular viewer:
//   {
//     type: "compound",
//     compound_id: "C0004",
//     elements: ["E00024", "E00052", ...],
//     source: "molecule"
//   }
//
// If type is not set, we infer from keys / id prefixes.
// ============================================================================

import React, { useMemo } from "react";

function asArray(x) {
  if (!x) return [];
  if (Array.isArray(x)) return x;
  return [x];
}

// ---------------------------------------------------------------------------
// ELEMENT PROFILE
// ---------------------------------------------------------------------------
function getElementProfile(results, active) {
  if (!results || !active) return null;

  const elId =
    active.element ||
    active.id ||
    (typeof active === "string" ? active : null);
  if (!elId) return null;

  const descRaw = results.element_descriptions || {};
  const descList = Array.isArray(descRaw)
    ? descRaw
    : Object.keys(descRaw).map((k) => ({ id: k, ...descRaw[k] }));

  const info =
    descList.find(
      (e) =>
        e.element === elId ||
        e.id === elId ||
        e.code === elId ||
        String(e.element_id) === String(elId)
    ) || null;

  const elemsTable = Array.isArray(results.hilbert_elements)
    ? results.hilbert_elements
    : [];

  const contexts = [];
  for (const row of elemsTable) {
    const rid =
      row.element || row.ELEMENT || row.id || row.code || row.element_id;
    if (String(rid) !== String(elId)) continue;
    const span = row.context || row.span || row.text || row.raw || "";
    if (span && contexts.length < 4) contexts.push(span);
  }

  if (!info && !contexts.length) return null;

  const label =
    info?.label || info?.token || info?.name || active.label || elId;

  const entropy =
    info?.metrics?.mean_entropy ??
    info?.mean_entropy ??
    info?.entropy ??
    active.entropy ??
    null;
  const coherence =
    info?.metrics?.mean_coherence ??
    info?.mean_coherence ??
    info?.coherence ??
    active.coherence ??
    null;

  const regime =
    info?.regime_profile ||
    info?.regime ||
    (info && {
      info: info.info_score,
      mis: info.misinfo_score,
      dis: info.disinfo_score,
    }) ||
    null;

  return {
    type: "element",
    id: elId,
    label,
    entropy,
    coherence,
    regime,
    contexts,
    raw: info,
  };
}

// ---------------------------------------------------------------------------
// COMPOUND PROFILE
// ---------------------------------------------------------------------------
function getCompoundProfile(results, active) {
  if (!results || !active) return null;

  const cidGuess =
    active.compound_id ||
    active.id ||
    active.compound ||
    (typeof active === "string" ? active : null);

  if (!cidGuess) return null;
  const cid = String(cidGuess);

  // informational_compounds may be array or map
  let comp = null;
  const ic = results.informational_compounds;
  if (Array.isArray(ic)) {
    comp =
      ic.find(
        (c) =>
          c.id === cid ||
          c.compound_id === cid ||
          c.code === cid ||
          String(c.cid) === cid
      ) || null;
  } else if (ic && typeof ic === "object" && ic[cid]) {
    comp = { id: cid, ...ic[cid] };
  }

  // compound_metrics may be { compounds: [...] } or plain array
  let metrics = null;
  const cm = results.compound_metrics;
  const cmList = Array.isArray(cm?.compounds)
    ? cm.compounds
    : Array.isArray(cm)
    ? cm
    : null;
  if (cmList) {
    metrics =
      cmList.find(
        (c) =>
          c.id === cid ||
          c.compound_id === cid ||
          c.code === cid ||
          String(c.cid) === cid
      ) || null;
  }

  if (!comp && !metrics) return null;

  const elements =
    asArray(
      comp?.elements ||
        comp?.member_elements ||
        comp?.nodes ||
        comp?.element_ids
    ).map(String) || asArray(active.elements).map(String);

  const numElements =
    metrics?.num_elements ??
    comp?.num_elements ??
    (elements.length || undefined);

  const numBonds =
    metrics?.num_bonds ?? comp?.num_bonds ?? comp?.bonds?.length ?? null;

  const stability =
    metrics?.compound_stability ??
    metrics?.stability ??
    comp?.stability ??
    null;

  const meanTemp =
    metrics?.mean_temperature ?? comp?.mean_temperature ?? null;

  const regime =
    comp?.regime_profile ||
    metrics?.regime_profile || {
      info: metrics?.info ?? 0,
      mis: metrics?.mis ?? 0,
      dis: metrics?.dis ?? 0,
    };

  return {
    type: "compound",
    id: cid,
    elements,
    numElements,
    numBonds,
    stability,
    meanTemp,
    regime,
    rawComp: comp,
    rawMetrics: metrics,
  };
}

// ---------------------------------------------------------------------------
// RENDER HELPERS
// ---------------------------------------------------------------------------
function formatRegime(regime) {
  if (!regime) return null;
  const info = Number(regime.info || regime.info_frac || 0);
  const mis = Number(regime.mis || regime.misinfo || 0);
  const dis = Number(regime.dis || regime.disinfo || 0);

  const sum = info + mis + dis || 1;
  const pct = (x) => ((x / sum) * 100).toFixed(1);

  return {
    info: pct(info),
    mis: pct(mis),
    dis: pct(dis),
  };
}

function RegimeBadges({ regime }) {
  const r = formatRegime(regime);
  if (!r) return null;
  return (
    <div
      style={{
        display: "flex",
        gap: 8,
        fontSize: 10,
        marginTop: 2,
      }}
    >
      <span style={{ color: "#10b981" }}>Info {r.info}%</span>
      <span style={{ color: "#f97316" }}>Mis {r.mis}%</span>
      <span style={{ color: "#ef4444" }}>Dis {r.dis}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// COMPONENT
// ---------------------------------------------------------------------------
export default function ThesisAdvisorPanel({
  results,
  activeElement,
  appendLog,
}) {
  // Decide what we're focusing on
  const focus = useMemo(() => {
    if (!activeElement) return null;

    const t = activeElement.type;

    // Explicit compound type
    if (t === "compound") {
      return getCompoundProfile(results, activeElement) || null;
    }

    // Infer compound from id prefix if not typed
    const idish =
      activeElement.compound_id ||
      activeElement.id ||
      activeElement.element;
    if (idish && /^C\d+/i.test(String(idish))) {
      const cp = getCompoundProfile(results, {
        ...activeElement,
        compound_id: idish,
      });
      if (cp) return cp;
    }

    // Otherwise treat as element
    return getElementProfile(results, activeElement) || null;
  }, [results, activeElement]);

  // Log on change (light, no infinite loops)
  React.useEffect(() => {
    if (!focus || !appendLog) return;
    if (focus.type === "compound") {
      appendLog(
        `[Thesis] Focus set to compound ${focus.id} with ${focus.numElements ||
          "?"} elements.`
      );
    } else if (focus.type === "element") {
      appendLog(
        `[Thesis] Focus set to element ${focus.id} (${focus.label}).`
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.id, focus?.type]);

  // No focus yet
  if (!focus) {
    return (
      <div
        className="hilbert-panel-body"
        style={{
          padding: 10,
          color: "#8b949e",
          fontSize: 11,
        }}
      >
        Select an element from the Periodic Table or a compound from the
        Molecular map to see an interpretive summary.
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render COMPOUND view
  // -------------------------------------------------------------------------
  if (focus.type === "compound") {
    const regime = formatRegime(focus.regime);
    return (
      <div
        className="hilbert-panel-body"
        style={{
          display: "flex",
          flexDirection: "column",
          padding: 10,
          gap: 8,
          color: "#e6edf3",
          fontSize: 11,
        }}
      >
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            marginBottom: 2,
          }}
        >
          Thesis Advisor — Compound Focus
        </div>

        <div style={{ fontSize: 11, color: "#8b949e" }}>
          Interpret compounds in our informational chemistry system.
        </div>

        <div
          style={{
            padding: 8,
            borderRadius: 8,
            background: "#0d1117",
            border: "1px solid #30363d",
          }}
        >
          <div
            style={{
              fontWeight: 600,
              fontSize: 12,
              marginBottom: 4,
            }}
          >
            Compound {focus.id}
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div>
              <span style={{ color: "#8b949e" }}>Elements: </span>
              <span>
                {focus.numElements != null
                  ? focus.numElements
                  : focus.elements?.length || "—"}
              </span>
            </div>
            {focus.numBonds != null && (
              <div>
                <span style={{ color: "#8b949e" }}>Bonds: </span>
                <span>{focus.numBonds}</span>
              </div>
            )}
            {focus.stability != null && (
              <div>
                <span style={{ color: "#8b949e" }}>Stability: </span>
                <span>{focus.stability.toFixed
                  ? focus.stability.toFixed(3)
                  : focus.stability}</span>
              </div>
            )}
            {focus.meanTemp != null && (
              <div>
                <span style={{ color: "#8b949e" }}>Mean temperature: </span>
                <span>{focus.meanTemp.toFixed
                  ? focus.meanTemp.toFixed(3)
                  : focus.meanTemp}</span>
              </div>
            )}
          </div>

          {regime && (
            <div style={{ marginTop: 4 }}>
              <div
                style={{
                  fontSize: 10,
                  color: "#8b949e",
                  marginBottom: 2,
                }}
              >
                Regime profile (aggregate of member elements):
              </div>
              <RegimeBadges regime={focus.regime} />
            </div>
          )}

{focus.aggregate_keywords && (
  <div style={{ marginTop: 8 }}>
    <div style={{ fontSize: 10, color: "#8b949e" }}>Aggregate keywords:</div>
    <div style={{ fontSize: 10, color: "#58a6ff", marginTop: 2 }}>
      {focus.aggregate_keywords.slice(0, 15).join(", ")}
    </div>
  </div>
)}

{focus.context_examples && focus.context_examples.length > 0 && (
  <div
    style={{
      marginTop: 10,
      padding: 8,
      borderRadius: 8,
      background: "#020817",
      border: "1px solid #30363d",
    }}
  >
    <div
      style={{
        fontSize: 10,
        fontWeight: 600,
        marginBottom: 4,
        color: "#8b949e",
      }}
    >
      Sentences where most members co-occur:
    </div>
    {focus.context_examples.slice(0, 5).map((c, i) => (
      <div
        key={i}
        style={{
          marginBottom: 4,
          padding: 4,
          borderRadius: 4,
          background: "#02040a",
          border: "1px solid #161b22",
          fontSize: 10,
          color: "#9ca3af",
        }}
      >
        {c}
      </div>
    ))}
  </div>
)}

        </div>

        <div
          style={{
            marginTop: 4,
            padding: 8,
            borderRadius: 8,
            background: "#020817",
            border: "1px dashed #30363d",
            fontSize: 10,
            color: "#8b949e",
          }}
        >
          This compound captures a tightly coupled informational motif. Use it
          to reason about how these elements co-occur and whether they behave as
          a stable narrative unit or a volatile cluster of claims.
        </div>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Render ELEMENT view (default)
  // -------------------------------------------------------------------------
  const { id, label, entropy, coherence, regime, contexts } = focus;

  return (
    <div
      className="hilbert-panel-body"
      style={{
        display: "flex",
        flexDirection: "column",
        padding: 10,
        gap: 8,
        color: "#e6edf3",
        fontSize: 11,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          marginBottom: 2,
        }}
      >
        Thesis Advisor — Element Focus
      </div>

      <div style={{ fontSize: 11, color: "#8b949e" }}>
        Interpret your corpus as an informational chemistry system.
      </div>

      <div
        style={{
          padding: 8,
          borderRadius: 8,
          background: "#0d1117",
          border: "1px solid #30363d",
        }}
      >
        <div
          style={{
            fontWeight: 600,
            fontSize: 12,
            marginBottom: 2,
          }}
        >
          Focal element:{" "}
          <span style={{ color: "#58a6ff" }}>
            {id} — {label}
          </span>
        </div>

        <div
          style={{
            display: "flex",
            gap: 16,
            flexWrap: "wrap",
            marginTop: 2,
          }}
        >
          {entropy != null && (
            <div>
              <span style={{ color: "#8b949e" }}>Entropy: </span>
              <span>{entropy.toFixed ? entropy.toFixed(3) : entropy}</span>
            </div>
          )}
          {coherence != null && (
            <div>
              <span style={{ color: "#8b949e" }}>Coherence: </span>
              <span>
                {coherence.toFixed ? coherence.toFixed(3) : coherence}
              </span>
            </div>
          )}
        </div>

        {regime && (
          <>
            <div
              style={{
                marginTop: 4,
                fontSize: 10,
                color: "#8b949e",
              }}
            >
              Regime profile:
            </div>
            <RegimeBadges regime={regime} />
          </>
        )}
      </div>

      {contexts && contexts.length > 0 && (
        <div
          style={{
            marginTop: 4,
            padding: 8,
            borderRadius: 8,
            background: "#020817",
            border: "1px solid #30363d",
          }}
        >
          <div
            style={{
              fontSize: 10,
              fontWeight: 600,
              marginBottom: 4,
              color: "#8b949e",
            }}
          >
            Context examples
          </div>
          {contexts.map((c, i) => (
            <div
              key={i}
              style={{
                marginBottom: 4,
                padding: 4,
                borderRadius: 4,
                background: "#02040a",
                border: "1px solid #161b22",
                fontSize: 10,
                color: "#9ca3af",
              }}
            >
              {c}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
