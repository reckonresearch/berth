#!/usr/bin/env bash
# Run ONCE per fresh pod, BEFORE starting the vLLM server.
# Fixes the environment traps found during P0 calibration:
#   - pyairports 0.0.1 is malformed (pip "satisfied" but unimportable); vLLM's
#     guided-decoding imports it at server startup, so it must exist first.
#   - HF Xet transfer backend is flaky; disable it for reliable downloads.
#   - HF cache must live on the large /workspace volume, not the 20GB container disk.
set -euo pipefail

# 1. pyairports stub (interface outlines/guided-decoding needs; empty list is fine)
python - <<'PYSTUB'
import os, site
sp = site.getsitepackages()[0] if hasattr(site, "getsitepackages") else "/usr/local/lib/python3.11/dist-packages"
d = os.path.join(sp, "pyairports")
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "__init__.py"), "w").close()
with open(os.path.join(d, "airports.py"), "w") as f:
    f.write("AIRPORT_LIST = []\n")
# verify
from importlib import invalidate_caches; invalidate_caches()
import pyairports.airports as a
assert a.AIRPORT_LIST == [], "pyairports stub failed"
print("pyairports stub OK")
PYSTUB

# 2. HF environment (export these in your shell too; this script only sets them
#    for its own process, so the ECHO block below reminds you to export them).
echo ""
echo "Now export these in your server shell BEFORE launching vLLM:"
echo "  export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1"
echo "  pip uninstall -y hf_xet   # optional; disabling via env is enough"
echo ""
echo "setup_env.sh complete. Start the server, then run bench/p0_run.sh."
