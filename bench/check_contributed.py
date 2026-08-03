"""Gate on contributed traces: measured only, provenance explicit.

The corpus is the asset. Its value is entirely that every cell in it was
observed on hardware, so a single simulated file admitted by accident destroys
more than it adds, and destroys it invisibly.

Two rules, and the second is stricter than the loader on purpose:

  1. No record may carry source == "mock".
  2. Every record must carry source explicitly. bench.sounding's loader
     back-fills "measured" on schema-1 files, which is right for this repo's
     own pre-existing traces and wrong for a file from a stranger: we know
     ours are hardware, we do not know theirs. So contributions must be
     schema 2 and say so.

Usage:
    python -m bench.check_contributed data/contributed/
Exit 0 clean, 1 rejected, 2 unusable input.
"""

import argparse
import json
import os
import sys


def check_file(path):
    """Return a list of complaints, each as (line_number, message)."""
    problems, notes = [], []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append((i, f"not valid JSON: {exc}"))
                continue
            schema = d.get("schema")
            source = d.get("source")
            if source is None:
                problems.append((i, "no 'source' field. Contributions must be "
                                    "schema 2 and state their provenance "
                                    "explicitly; re-run with a current "
                                    "bench.sounding."))
            elif source == "mock":
                problems.append((i, "source is 'mock'. Mock traces exercise the "
                                    "pipeline; they are not measurements and "
                                    "cannot enter the corpus."))
            elif source != "measured":
                problems.append((i, f"unknown source {source!r}"))
            if schema is not None and schema < 2:
                problems.append((i, f"schema v{schema}; contributions require v2"))
            prov = d.get("silicon_provenance")
            if prov == "mock":
                problems.append((i, "silicon_provenance is 'mock'"))
            elif prov == "self_reported":
                # Not a rejection. The corpus accepts self-reported cells,
                # because a remote endpoint genuinely cannot be inspected. But
                # the index should be able to tell the two apart, and a
                # reviewer should see it before merging rather than after.
                notes.append((i, "silicon_provenance is 'self_reported': the "
                                 "hardware identity was asserted, not captured. "
                                 "Accepted, but worth confirming with the "
                                 "contributor that the box matches the label."))
    return problems, notes


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bench.check_contributed",
        description="Reject mock or unlabelled traces in contributed data.")
    p.add_argument("paths", nargs="+", help="files or directories to scan")
    args = p.parse_args(argv)

    files = []
    for path in args.paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.endswith(".jsonl")]
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"no such path: {path}", file=sys.stderr)
            return 2

    if not files:
        print("no .jsonl files found; nothing to check")
        return 0

    rejected = 0
    for f in files:
        problems, notes = check_file(f)
        for line_no, msg in notes:
            print(f"note     {f} line {line_no}: {msg}")
        if problems:
            rejected += 1
            print(f"REJECTED {f}")
            for line_no, msg in problems[:10]:
                print(f"   line {line_no}: {msg}")
            if len(problems) > 10:
                print(f"   ... and {len(problems) - 10} more")
        else:
            print(f"ok       {f}")

    if rejected:
        print(f"\n{rejected} of {len(files)} files rejected. "
              "Every contributed cell must be a hardware measurement that says "
              "so on every line.")
        return 1
    print(f"\n{len(files)} files clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
