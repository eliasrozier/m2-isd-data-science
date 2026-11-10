#!/usr/bin/env bash
# Small data subsets for the dry run (~1.5 GB total, ~5-10 min on Colab/Snowflake).
# Usage: bash dryrun/fetch_data.sh [DATA_DIR]      (default: data/)
# Each block is independent: a failed download only skips the steps that need it.
set -u
TRACK="$(cat "$(dirname "$0")/TRACK" 2>/dev/null || echo all)"   # vision | speech | all
want(){ [ "$TRACK" = all ] || [ "$TRACK" = "$1" ]; }
D="${1:-data}"; mkdir -p "$D"; cd "$D"
ok(){ echo "  [ok] $1"; }; ko(){ echo "  [FAILED] $1 -- the related smoke steps will be skipped"; }

if want speech; then
echo "== FSDD (speech classical, 10 MB)"
[ -d fsdd ] || git clone -q --depth 1 https://github.com/Jakobovski/free-spoken-digit-dataset.git fsdd && ok fsdd || ko fsdd

fi
if want vision; then
echo "== Short street video for the video pipeline (OpenCV sample, no ground truth)"
[ -f vtest.avi ] || wget -q https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi && ok vtest.avi || ko vtest.avi

echo "== COCO val2017: annotations (241 MB zip) + 50 images fetched one by one"
if [ ! -f coco/annotations/instances_val2017.json ]; then
  mkdir -p coco && wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O coco/ann.zip \
    && unzip -q -o coco/ann.zip "annotations/instances_val2017.json" -d coco && rm coco/ann.zip && ok annotations || ko annotations
fi
if [ -f coco/annotations/instances_val2017.json ]; then
  mkdir -p coco/val2017
  python - <<'PY'
import json, random, urllib.request, pathlib
a = json.load(open("coco/annotations/instances_val2017.json"))
random.seed(0); imgs = random.sample(a["images"], 50)
for im in imgs:
    p = pathlib.Path("coco/val2017") / im["file_name"]
    if not p.exists():
        urllib.request.urlretrieve(im["coco_url"], p)
# shrink the annotation file to these 50 images so evaluation is consistent and fast
ids = {im["id"] for im in imgs}
a["images"] = imgs; a["annotations"] = [x for x in a["annotations"] if x["image_id"] in ids]
json.dump(a, open("coco/annotations/instances_val2017.json", "w"))
print("  [ok] 50 COCO images")
PY
fi

fi
if want speech; then
echo "== Reference on-device streaming model (sherpa-onnx zipformer, 20M params, ~130 MB) -> ../models"
M="$(cd "$(dirname "$0")/.." && pwd)/models"; mkdir -p "$M"
[ -d "$M/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17" ] || (cd "$M" && wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2 && tar xjf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2 && rm sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2) && ok reference-model || ko reference-model
echo "== LibriSpeech test-clean (346 MB) + pruned 3-gram LM (~34 MB)"
[ -d LibriSpeech/test-clean ] || (wget -q https://www.openslr.org/resources/12/test-clean.tar.gz && tar xzf test-clean.tar.gz && rm test-clean.tar.gz) && ok librispeech || ko librispeech
[ -f lm/3-gram.pruned.1e-7.arpa ] || (mkdir -p lm && wget -q https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz -O lm/lm.gz && gunzip -c lm/lm.gz | tr '[:upper:]' '[:lower:]' > lm/3-gram.pruned.1e-7.arpa && rm lm/lm.gz) && ok arpa || ko arpa
# (lower-cased because wav2vec2 labels are lower-cased in speech/ctc_lm.py)

fi
if want vision; then
echo "== DAVIS 2017 trainval 480p (~800 MB) -- needed only for the SAM2 step"
[ -d DAVIS ] || (wget -q https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip && unzip -q DAVIS-2017-trainval-480p.zip && rm DAVIS-2017-trainval-480p.zip) && ok davis || ko davis
fi
du -sh . 2>/dev/null
