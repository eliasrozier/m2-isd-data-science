# Decision log

One entry per scientific choice, at least one per milestone. Keep entries short and factual: the oral
exam asks you to defend them, including the alternatives you rejected.

## D1 — <decision in a few words>   (milestone: d1)
- Question:
- Alternatives:
- Protocol:
- Evidence:
- Decision:
- Would change our mind if:



## Decision — Day 3: Deployment under a 200 ms median finish-delay budget

We evaluated seven streaming configurations on the same 20 LibriSpeech utterances, comparing streaming WER and median finish delay.

The reference FastConformer 114M INT8 achieved the lowest streaming WER (2.1%) and the lowest median finish delay (150 ms), with no observed processing backlog.

It dominates all six offline re-decoding configurations on these two objectives and is the only candidate satisfying our 200 ms median finish-delay budget.

We therefore select FastConformer 114M INT8 as our deployment candidate.

This decision is limited to the evaluated utterances and hardware. Further validation should include more recordings, noisy conditions, memory consumption and latency variability.