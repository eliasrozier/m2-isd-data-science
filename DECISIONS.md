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

## Day 3 Afternoon — Live ASR Demonstration

We integrated our Day 2 CTC decoder with a 3-gram language model into the provided ASR demonstration framework. We compared it with Whisper Tiny, wav2vec2 Greedy, and the FastConformer 114M INT8 reference model.

The first demonstration was conducted in a noisy environment. The offline replay produced a WER of 0% for Whisper Tiny and 38% for both CTC configurations. FastConformer, tested through live microphone input, achieved a WER of 25%, with a first-partial delay of 3.024 seconds and a final latency of 149 milliseconds.

These results are preliminary because the live and replay conditions were not identical. They cannot support a direct ranking of the models.

The experiment confirmed that FastConformer can provide partial transcriptions while speech is still being recorded, whereas our offline CTC backend returns its transcription after the recording ends.

We plan to repeat the experiment in a quieter environment, using the same audio recording across all backends to ensure a fair comparison.

The larger Day 3 benchmark remains the primary evidence for our deployment decision.