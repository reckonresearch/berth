"""`pilot` as its own command.

berth is the estimator and pilot is the agent. Reaching the agent through
`berth pilot` reads as the estimator's subcommand, which is the wrong shape:
one product does not live inside another's command surface.

Same package, same code, second entry point. `berth pilot` still works, so
nothing breaks for anyone who scripted against it.
"""

import sys

from berth.cli import build_parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Route straight to the pilot subcommand so `pilot --classes x` works
    # rather than requiring `pilot pilot --classes x`.
    if not argv or argv[0] in ("-h", "--help"):
        argv = ["pilot", "--help"] if argv else ["pilot", "--help"]
    elif argv[0] != "pilot":
        argv = ["pilot", *argv]
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
