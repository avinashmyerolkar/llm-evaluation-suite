# Golden datasets

Every eval in this repo is driven by a **hand-authored golden set**: a fixed set of inputs plus
whatever ground truth that eval needs. Because the set never changes, a score only moves when the
system changes, not when someone happens to test it with different questions.

> **The real golden sets are private and not in this repo.** [`samples/`](samples) contains
> **dummy data** about a made-up product, with the exact schema, tags and labels the evals expect.
> Each eval loads `goldens/<name>.json` if it exists (git-ignored) and otherwise falls back to
> `goldens/samples/<name>.json`. Entries marked `<DUMMY>` describe the kind of attack instead of
> including a working one.

| File | Used by | Ground truth fields |
|---|---|---|
| `retriever_golden.json` | `eval_retriever` | `query`, `ideal_answer` (recall/precision are measured against it) |
| `faithfulness_dataset.json` | `eval_generator`, `eval_rag_pipeline` | `query`, `ideal_context` (known-good chunks) |
| `correctness_goldens.json` | `eval_application` | `question`, `ideal_answer` |
| `leakage_goldens.json` | `eval_leakage` | `input`, `subtype` (`prompt` / `course_content` / `pii`), `technique`, `expected_action` |
| `toxicity_goldens.json` | `eval_toxicity` | `input`, `case_type` (`attack` / `benign`), `technique` |
| `scope_goldens.json` | `eval_scope_safety` | `input`, `expected_action`, `success_criteria` |

## How the sets are designed

- **Adversarial sets include benign controls.** `false_positive_guard` inputs are ones the system
  *should* answer. Without them, a guardrail that refuses everything would get a perfect score.
- **Attacks are tagged by technique** (`direct`, `roleplay`, `jailbreak`, `mixed`,
  `incremental_extraction`, `transformation`, `raw_retrieval`, `paste_injection`,
  `credential_exposure`, …), so a failure points at the attack vector that got through.
- **Expected behaviour is three-valued:** `ANSWER` / `DECLINE` / `PARTIAL`. `PARTIAL` covers mixed
  prompts where the right response is to answer the legitimate part and refuse the rest.
- **Credentials and identifiers are always synthetic** (`sk-test-DUMMY-…`, `example.com`).

## Building your own

1. Index your corpus (`uv run python -m src.retriever`).
2. Dump the chunks with `uv run python -m src.export_chroma_chunks`. This writes
   `goldens/chunks_dump.json`, which is git-ignored.
3. Hand-pick `ideal_context` chunks and write `ideal_answer`s. Save them as `goldens/<name>.json`
   using the schemas in `samples/`.
