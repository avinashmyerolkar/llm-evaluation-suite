# evals/eval_online.py
"""
Online eval: score LIVE production traces, reference-free.

Offline evals run on a golden set before release. Online evals run AFTER
release, on whatever real users actually ask -- so there is no answer key. The
RAG triad works here because it is reference-free: it judges the answer against
the query and the retrieved context the trace already carries.

    RagPipeline.invoke (@traceable)  ->  LangSmith trace {query, context, answer}
    this worker                      ->  sample traces, judge the triad,
                                         write scores back as trace FEEDBACK

Feedback lives on the trace itself, so LangSmith can chart faithfulness over
time, filter low-scoring traces, and turn bad ones into new golden cases.

Design points:
  - same three metrics as the offline triad (eval_rag_pipeline.py), so online
    and offline numbers are directly comparable
  - deterministic sampling on the run UUID -> a trace is either always or never
    sampled, across processes and cron invocations (cost stays bounded)
  - per-key dedup -> re-running the worker never re-judges (or re-bills) a metric
  - bounded lookback window -> each pass reads recent traces, not all history
  - one metric failing never sinks the other two for the same trace

    python -m evals.eval_online        # poll forever (dev); use cron in production
"""

import os
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langsmith import Client

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

load_dotenv()

# Must match the project the app traces into (LANGSMITH_PROJECT in .env).
PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-online-evals")
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7
SAMPLE_RATE = 1.0        # fraction of traces to judge. 1.0 = all (dev); ~0.1-0.3 in production,
                         # since each trace costs 3 metrics x several judge calls
LOOKBACK_HOURS = 24      # only consider traces from this window on each pass
POLL_SECONDS = 60

client = Client()


def _sampled(run):
    """Deterministic per-run sampling: same run always decided the same way.

    Uses the run's UUID, not hash(): Python salts str hashes per process, which
    would re-roll the decision on every cron invocation.
    """
    return run.id.int % 100 < SAMPLE_RATE * 100


def _existing_keys(run):
    """Which feedback keys this run already has — so we never re-judge a key."""
    fb = client.list_feedback(run_ids=[run.id])
    return {f.key for f in fb}


def score_recent_traces():
    """One pass: read recent RagPipeline traces, score any missing triad metric, push feedback."""
    runs = client.list_runs(
        project_name=PROJECT,
        is_root=True,             # RagPipeline root only, not child spans
        run_type="chain",
        filter='eq(name, "RagPipeline")',
        start_time=datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS),
    )

    scored = 0
    for run in runs:
        if not _sampled(run):
            continue

        outputs = run.outputs or {}
        answer = outputs.get("answer")
        context = outputs.get("context")     # post-rerank chunks — same field offline uses
        query = (run.inputs or {}).get("query", "")

        if not answer or not context:
            continue                          # nothing to judge

        already = _existing_keys(run)

        # each metric: (feedback key, metric object, which inputs it needs)
        # built fresh per run so scores/reasons don't leak between traces
        jobs = [
            ("faithfulness",
             FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),

            ("answer_relevancy",
             AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer)),

            ("contextual_relevancy",
             ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
             dict(input=query, actual_output=answer, retrieval_context=context)),
        ]

        for key, metric, tc_kwargs in jobs:
            if key in already:
                continue                      # per-key dedup: skip only what's already scored

            try:
                metric.measure(LLMTestCase(**tc_kwargs))
                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason,
                )
                scored += 1
            except Exception as e:
                # one metric failing shouldn't sink the other two for this trace
                print(f"[{key}] failed on run {run.id}: {e}")

    print(f"[online] pass complete: {scored} new scores written to '{PROJECT}'")


if __name__ == "__main__":
    # live-in-a-terminal version. For production, delete the loop and let cron
    # call score_recent_traces() on a schedule instead.
    while True:
        score_recent_traces()
        time.sleep(POLL_SECONDS)
