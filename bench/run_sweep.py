"""Deprecated module path. The meter is now `sounding`.

`bench.run_sweep` is retained as a shim so existing commands keep working.
Prefer `python -m bench.sounding`.
"""
from bench.sounding import *  # noqa: F401,F403
from bench.sounding import main

if __name__ == "__main__":
    import warnings
    warnings.warn("bench.run_sweep is renamed to bench.sounding; update your command",
                  DeprecationWarning, stacklevel=2)
    main()
