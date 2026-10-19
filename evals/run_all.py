"""
evals/run_all.py — run every quality + safety layer against ONE pipeline and
save the per-metric summaries as a labelled snapshot.

    python -m evals.run_all baseline
    python -m evals.run_all candidate --compare baseline

Snapshots hold numbers only (pass rate + score stats per metric), never model
outputs or golden content, so they are safe to commit. Comparing two snapshots
shows which metric moved after a prompt, model or retrieval change.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from src.rag_pipeline import RagPipeline
from evals import (
    eval_retriever,
    eval_generator,
    eval_rag_pipeline,
    eval_application,
    eval_leakage,
    eval_toxicity,
    eval_scope_safety,
)
from evals.harness import print_summary

RESULTS_DIR = "results"


def run_all():
    rag = RagPipeline()   # load the store + reranker once, share it across layers
    layers = {
        "retriever":    lambda: eval_retriever.run(rag.retriever),
        "generator":    eval_generator.run,
        "rag_pipeline": lambda: eval_rag_pipeline.run(rag),
        "application":  lambda: eval_application.run(rag),
        "leakage":      lambda: eval_leakage.run(rag),
        "toxicity":     lambda: eval_toxicity.run(rag),
        "scope":        lambda: eval_scope_safety.run(rag),
    }
    return {name: fn() for name, fn in layers.items()}


def compare(current, previous):
    """Print the pass-rate / avg-score delta for every metric in both snapshots."""
    print("\n" + "=" * 72)
    print(f"{'layer / metric':<40} {'pass rate':>14} {'avg score':>14}")
    print("=" * 72)
    for layer, metrics in current.items():
        for name, s in metrics.items():
            p = previous.get(layer, {}).get(name)
            if not p:
                continue
            d_pass = s["pass_rate"] - p["pass_rate"]
            d_avg = s["avg_score"] - p["avg_score"]
            print(f"{layer + ' / ' + name:<40} {p['pass_rate']:5.0f}→{s['pass_rate']:<4.0f}({d_pass:+4.0f})"
                  f" {p['avg_score']:5.2f}→{s['avg_score']:.2f} ({d_avg:+.2f})")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="snapshot name, e.g. baseline")
    ap.add_argument("--compare", help="label of a previous snapshot to diff against")
    args = ap.parse_args()

    results = run_all()
    for layer, summary in results.items():
        print_summary(layer, summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    snapshot = {
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layers": results,
    }
    with open(f"{RESULTS_DIR}/{args.label}.json", "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nsaved {RESULTS_DIR}/{args.label}.json")

    if args.compare:
        with open(f"{RESULTS_DIR}/{args.compare}.json") as f:
            compare(results, json.load(f)["layers"])


if __name__ == "__main__":
    main()
