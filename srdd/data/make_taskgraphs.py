#!/usr/bin/env python3
"""
Batch pre-generate SRDD task_graph.json files using the Qwen graph planner.

Reads a SRDD CSV file (iid_train / iid_test / full SRDD.csv) and generates
task_graph.json + sample.json for each sample, storing them under --output-root.

Output layout (compatible with train_mole_srdd_hgrpo.py --taskgraph-root):
  <output-root>/<Category>/<Name>/task_graph.json
  <output-root>/<Category>/<Name>/sample.json   (← required by _load_taskgraph_index)

Usage:
  python pregenerate_srdd_taskgraphs.py \
    --csv  /path/to/iid_train.csv \
    --output-root /path/to/taskgraph/srdd \
    --graph-model-name Qwen/Qwen3-4B-Instruct-2507 \
    --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Repo path setup ────────────────────────────────────────────────────────
_SCRIPT = Path(__file__).resolve()
_REPO   = _SCRIPT.parents[2]            # Task-Action-MAS/
for _p in [str(_REPO)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from morse.taskgraph.generator import GraphPlanner  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_") or "item"


def _load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["Category"].strip(), r["Name"].strip(), r["Description"].strip())
            for r in rows if r.get("Name", "").strip() and r.get("Description", "").strip()]


def _build_planner(model_name: str, gpu: int, torch_dtype: str) -> GraphPlanner:
    dtype = getattr(torch, torch_dtype, torch.bfloat16)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    max_memory = None
    if n_gpu > 0:
        total = torch.cuda.get_device_properties(gpu).total_memory
        allowed = int(0.90 * total)
        max_memory = {i: (allowed if i == gpu else 0) for i in range(n_gpu)}

    model_kwargs = {"torch_dtype": dtype}
    if max_memory:
        model_kwargs["max_memory"] = max_memory

    return GraphPlanner(
        provider="hf",
        model_name=model_name,
        max_new_tokens=2048,
        temperature=0.0,
        model_kwargs=model_kwargs,
        device=gpu,
        device_map="auto",
        allow_fallback=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Batch pre-generate SRDD task graphs.")
    p.add_argument("--csv", type=Path, required=True,
                   help="SRDD CSV file (iid_train.csv / iid_test.csv / SRDD.csv).")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Root directory to save generated task_graph.json files.")
    p.add_argument("--graph-model-name", type=str,
                   default="Qwen/Qwen3-4B-Instruct-2507",
                   help="HF model ID or local path for the graph generation LLM.")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index for the graph model.")
    p.add_argument("--torch-dtype", type=str, default="bfloat16")
    p.add_argument("--retries", type=int, default=3,
                   help="Max generation retries per sample.")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip samples that already have a task_graph.json.")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = p.parse_args()

    samples = _load_csv(args.csv)
    print(f"[pregen] Loaded {len(samples)} samples from {args.csv}", flush=True)
    print(f"[pregen] Output root: {args.output_root}", flush=True)
    print(f"[pregen] Graph model: {args.graph_model_name} on GPU {args.gpu}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Count already done
    done = sum(
        1 for cat, name, _ in samples
        if (args.output_root / _sanitize(cat) / _sanitize(name) / "task_graph.json").exists()
    )
    print(f"[pregen] Already done: {done}/{len(samples)}", flush=True)

    if done == len(samples) and args.skip_existing:
        print("[pregen] All samples already generated. Nothing to do.", flush=True)
        return

    print(f"[pregen] Loading graph model...", flush=True)
    planner = _build_planner(args.graph_model_name, args.gpu, args.torch_dtype)
    print(f"[pregen] Model loaded.", flush=True)

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for idx, (cat, name, desc) in enumerate(samples):
        sample_dir = args.output_root / _sanitize(cat) / _sanitize(name)
        graph_path = sample_dir / "task_graph.json"
        sample_json = sample_dir / "sample.json"

        if args.skip_existing and graph_path.exists():
            n_skip += 1
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        # Always (re)write sample.json so _load_taskgraph_index can find it
        sample_json.write_text(
            json.dumps({"category": cat, "name": name, "description": desc},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        t0 = time.time()
        graph = None
        last_err: Optional[Exception] = None
        for attempt in range(1, args.retries + 1):
            try:
                graph = planner.generate(name, desc)
                if getattr(planner, "used_fallback", False):
                    last_err = planner.last_error or RuntimeError("used heuristic fallback")
                    print(f"  [attempt {attempt}] fallback used, retrying...", flush=True)
                    graph = None
                    continue
                break
            except Exception as exc:
                last_err = exc
                print(f"  [attempt {attempt}] error: {exc}", flush=True)

        elapsed = time.time() - t0
        if graph is not None:
            graph.save_json(graph_path)
            n_ok += 1
            print(
                f"[{idx+1}/{len(samples)}] OK  {cat}/{name}  ({elapsed:.1f}s)",
                flush=True,
            )
        else:
            n_fail += 1
            fail_log = sample_dir / "graph_error.txt"
            fail_log.write_text(str(last_err), encoding="utf-8")
            print(
                f"[{idx+1}/{len(samples)}] FAIL {cat}/{name}  err={last_err}",
                flush=True,
            )

    print(
        f"\n[pregen] Done. ok={n_ok}  skip={n_skip}  fail={n_fail}  total={len(samples)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
