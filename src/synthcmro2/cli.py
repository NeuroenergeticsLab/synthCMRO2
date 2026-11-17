"""``synthcmro2 <command> ...``: one entry point for the model and figure commands.

model       check | train | select | predict | export   (config/model.yaml)
figures     Figures 2-4 and S2 with their statistics tables  (config/figures.yaml)
pls         mean-centered PLS of calculation versus rest (Figure 3b input; needs pyls)
benchmark   model errors versus the test-retest error of analytical CMRO2 (Table S2)
public-tables  shareable tables (group voxel maps, participant statistics) from the individual tables
crop-atlas  crop an atlas from the full 4-mm MNI grid to the training grid
"""

from __future__ import annotations

import sys


def _model(argv):
    from .model.commands import main

    return main(argv)


def _figures(name):
    def run(argv):
        from .figures import commands

        return getattr(commands, f"{name}_main")(argv)

    return run


COMMANDS = {
    "model": _model,
    "figures": _figures("figures"),
    "pls": _figures("pls"),
    "benchmark": _figures("benchmark"),
    "public-tables": _figures("public_tables"),
    "crop-atlas": _figures("crop_atlas"),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in {"-V", "--version"}:
        from . import __version__

        print(__version__)
        return 0
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    if argv[0] not in COMMANDS:
        print(__doc__)
        print(f"unknown command {argv[0]!r}")
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
