# Contributed traces

Measured traces only. Every line must carry `"source": "measured"` and
`"schema": 2`, which current `bench.sounding` writes for you.

    python -m bench.sounding --base-url http://localhost:8000 \
        --silicon <key> --model <key> --model-id <hf-id> --out traces.jsonl

Then open a pull request adding the file here. CI runs
`python -m bench.check_contributed` and rejects any file containing a mock
record or any record without an explicit `source`.

A trace carries no prompt content, only cell coordinates and two timing
numbers, so it is safe to share. Disputes that arrive with traces attached
outrank everything else.
