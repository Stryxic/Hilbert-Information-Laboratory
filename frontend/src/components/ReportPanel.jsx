import React from "react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";

export default function ReportPanel({ results }) {
  const pdfUrl = results?.hilbert_summary_pdf_url || results?.files?.hilbert_summary_pdf;
  const metrics = results?.compound_metrics || results?.compound_stats || {};
  const meta = results?.meta || {};
  const numCompounds = metrics.num_compounds ?? results?.num_compounds;
  const elements = meta.num_elements ?? results?.num_elements;

  return (
    <Card className="h-full flex flex-col bg-[#0d1117] border border-[#30363d] text-xs">
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#30363d]">
        <div>
          <div className="font-semibold text-[#e6edf3]">
            Run Summary & Reports
          </div>
          <div className="text-[9px] text-[#8b949e]">
            PDF overview, compound stats, and exports.
          </div>
        </div>
        {results?.run_folder && (
          <Button
            asChild
            size="xs"
            className="bg-[#21262d] hover:bg-[#30363d] text-[9px] px-2 py-1"
          >
            <a href={results.run_folder} target="_blank" rel="noreferrer">
              Open Folder
            </a>
          </Button>
        )}
      </div>

      <div className="px-3 py-2 grid grid-cols-3 gap-2 text-[9px] text-[#8b949e]">
        <div className="bg-[#151b23] rounded-md px-2 py-1 border border-[#30363d]">
          <div className="uppercase text-[7px]">Elements</div>
          <div className="text-[#e6edf3] text-xs">
            {elements ?? "—"}
          </div>
        </div>
        <div className="bg-[#151b23] rounded-md px-2 py-1 border border-[#30363d]">
          <div className="uppercase text-[7px]">Compounds</div>
          <div className="text-[#e6edf3] text-xs">
            {numCompounds ?? "—"}
          </div>
        </div>
        <div className="bg-[#151b23] rounded-md px-2 py-1 border border-[#30363d]">
          <div className="uppercase text-[7px]">Mean Stability</div>
          <div className="text-[#e6edf3] text-xs">
            {metrics.mean_stability != null
              ? metrics.mean_stability.toFixed(3)
              : "—"}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto px-3 pb-3">
        {pdfUrl ? (
          <iframe
            src={pdfUrl}
            title="Hilbert Summary"
            className="w-full h-full rounded border border-[#30363d]"
          />
        ) : (
          <div className="mt-4 text-[10px] text-[#8b949e]">
            The summary PDF will appear here after a successful run with
            exports enabled.
          </div>
        )}
      </div>
    </Card>
  );
}
