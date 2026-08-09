# AdvBench high-concurrency judge continuation

## Identity and objective

- Type: fresh judge-only continuation seeded from stopped run
  `2026-08-09_012344Z_advbench-jailbreak-concepts`.
- Objective: finish harmful-compliance judging without regenerating model outputs
  or recomputing concepts.
- Preserved inputs: all 520 generations, all 520 concept records, the concept
  distribution, and the first 220 judgments (zero recorded errors at stop).
- New output: a distinct dated run directory; the stopped source remains immutable.

## Changed method

- Judge model and schema remain unchanged: OpenAI Responses API,
  `gpt-5.6-luna`, low reasoning, structured output, `store=false`, prompt/schema v1.
- Evaluate only rows 220–519. Use a 160-worker `ThreadPoolExecutor`, one OpenAI
  client per worker thread, a 160-row checkpoint batch, three retries, exponential
  backoff, and random jitter. Calls within a batch run simultaneously.
- No GPU is requested. The Modal output Volume is committed at each completed
  batch. A controlled stop preserves the most recent completed batch.

## Hypotheses and expected results

1. Parallelism will improve judge throughput from about 1 response/second to at
   least 10 responses/second, completing the remaining 300 rows in roughly 1–4
   minutes. Provider rate limits may lower this; below 3 responses/second would
   disconfirm the throughput expectation.
2. At least 95% of the remaining rows will receive valid structured judgments.
   More than 5% terminal errors would indicate rate-limit, transport, or schema
   pressure from excessive concurrency.
3. Preserved judgments will be byte-for-byte reused, so the final ASR denominator
   and generation/concept artifacts should match a sequential continuation.

If results differ, first inspect HTTP status/error classes, retry counts, request
latency, rate-limit headers, duplicate/missing input indices, and structured-output
parsing. Suggested follow-ups are a 40/80/120 concurrency sweep, adaptive
rate-limit control, or replaying failed rows only. None will run without a new
card and explicit `proceed`.

## Artifacts, cost, and launch gate

The continuation produces the full original artifact set, including colored
prompt/generation cards, concept distribution, progress plots, paired JSONL,
aggregate JSON, and `RESULTS.md`. Expected incremental cost is only 300 OpenAI
judge calls plus a short CPU Modal container, likely a few minutes.

This materially changed card requires a new explicit **proceed** after it is
presented. The earlier signal preceded this card and does not authorize launch.
