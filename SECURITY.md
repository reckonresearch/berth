# Security
Report vulnerabilities privately to security@reckonresearch.com (do not open
a public issue). The harness executes no remote code and sends only
completion requests to the server URL you provide; trace files contain
shapes and timings only — never prompts or payloads.

## pilot

pilot needs read access to a configuration repository and write access to a
branch. It cannot merge, cannot write to a default branch, and cannot touch a
path that has not been declared in `.berth/classes.yaml`.

Those are enforced in code rather than by token scope. A scope is a promise
about configuration; this is a promise about code, in `berth/github.py`, and
each refusal is covered by a test. `merge()` exists solely to raise.

pilot never touches a request. There is no proxy, no gateway, and no traffic
path. The customer's own infrastructure moves the workload.
