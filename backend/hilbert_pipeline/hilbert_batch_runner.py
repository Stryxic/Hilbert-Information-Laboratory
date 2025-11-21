# hilbert_batch_runner.py - simple batched wrapper around run_pipeline

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from hilbert_orchestrator import PipelineSettings, run_pipeline


def run_batched_pipeline(
    corpus_root: Union[str, Path],
    results_root: Union[str, Path],
    batch_size: int = 5,
    use_native: bool = True,
) -> None:
    """
    Run the Hilbert pipeline in batches over the files in corpus_root.

    - corpus_root: directory with your input files (PDFs, txt, etc)
    - results_root: directory where a fused hilbert_elements.csv is written
    - batch_size: number of files per batch
    """

    corpus_root = Path(corpus_root).resolve()
    results_root = Path(results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in corpus_root.iterdir() if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise RuntimeError(f"No files found under {corpus_root}")

    batches: List[Sequence[Path]] = [
        files[i : i + batch_size] for i in range(0, len(files), batch_size)
    ]

    def emit(kind: str, payload: Dict[str, Any]) -> None:
        data = dict(payload)
        data.setdefault("batch_size", batch_size)
        print(json.dumps({"msg": "[batch]", "kind": kind, **data}))

    all_element_csvs: List[Path] = []

    for idx, batch in enumerate(batches):
        batch_dir = results_root / f"batch_{idx:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        tmp_corpus = batch_dir / "_corpus"
        if tmp_corpus.exists():
            shutil.rmtree(tmp_corpus)
        tmp_corpus.mkdir(parents=True)

        # copy this batch into its own mini-corpus
        for src in batch:
            shutil.copy2(src, tmp_corpus / src.name)

        settings = PipelineSettings(use_native=use_native, max_docs=None)

        emit("log", {"event": "batch_start", "batch_index": idx, "n_files": len(batch)})
        run_pipeline(
            corpus_dir=str(tmp_corpus),
            results_dir=str(batch_dir),
            settings=settings,
            emit=emit,
        )
        emit("log", {"event": "batch_end", "batch_index": idx})

        elements_csv = batch_dir / "hilbert_elements.csv"
        if elements_csv.exists():
            all_element_csvs.append(elements_csv)

    # Fuse elements from all batches into one CSV at results_root
    if all_element_csvs:
        import pandas as pd  # type: ignore

        frames = [pd.read_csv(p) for p in all_element_csvs]
        fused = pd.concat(frames, ignore_index=True)
        fused.to_csv(results_root / "hilbert_elements.csv", index=False)

        run_summary = {
            "run_id": f"batched_{int(time.time())}",
            "mode": "batched",
            "n_batches": len(batches),
            "batch_size": batch_size,
            "use_native": use_native,
            "n_elements": int(fused.shape[0]),
        }
        (results_root / "hilbert_run.json").write_text(
            json.dumps(run_summary, indent=2),
            encoding="utf-8",
        )
