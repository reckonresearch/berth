"""`sounding` as its own command.

The harness ran as `python -m bench.sounding`, which is a module path rather
than a product. sounding is one of three things this package provides and it
should be reachable the way the other two are.

bench.sounding.main() reads sys.argv itself and takes no arguments, so this
sets argv rather than passing it. Changing the harness signature would be the
cleaner fix and would also change how it is invoked everywhere else, including
in scripts an operator already has.
"""

import sys


def main(argv=None):
    from bench.sounding import main as sweep
    if argv is not None:
        sys.argv = ["sounding", *argv]
    return sweep()


if __name__ == "__main__":
    raise SystemExit(main())
