#!/usr/bin/env bash
# Must finish in < 10 min and rerun a SMALL version of your pipeline end to end. Adapt it at every milestone.
set -euo pipefail
python -m pytest -q tests/public
# add your own commands below, e.g. a 20-image benchmark or 5 utterances
