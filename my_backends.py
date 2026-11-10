"""Register YOUR systems here: they appear in the demo's backend list (python -m asr_demo.server).
Same contract as the provided ones (asr_demo/contract.py). Uncomment and adapt."""
# import numpy as np
# from asr_demo.backends import register, get
# from asr_demo.contract import OfflineBackend
#
# @register("student/ctc+3gram", "wav2vec2-base-100h + our KenLM decoder (alpha=0.5, beta=1.0)")
# class CTCWithLM(OfflineBackend):
#     def __init__(self):
#         ...                                   # load the model and build the decoder ONCE
#     def transcribe(self, audio: np.ndarray) -> str:
#         ...                                   # 16 kHz float32 in, lower-case words out
#
# A streaming wrapper around any registered offline backend (your day-3 policy):
# from speech.streaming_live import LocalAgreementBackend
# register("student/local-agreement(ctc+3gram)", "our LocalAgreement-2, 1 s step")(
#     lambda: LocalAgreementBackend(get("student/ctc+3gram"), step_s=1.0))
