#!/usr/bin/env python3
"""Compile the five handouts and the combined book with XeLaTeX/latexmk."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TARGETS = [
    "lecture01_convex", "lecture02_kkt", "lecture03_stochastic",
    "lecture04_optimizers", "lecture05_parameterization", "course_notes",
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="Target stems; default: all six PDFs.")
    parser.add_argument("--tex-support", type=Path, default=None,
                        help="Optional directory containing unpacked xeCJK/Fandol for a minimal TeX installation.")
    args = parser.parse_args()
    targets = args.targets or TARGETS
    if set(targets) - set(TARGETS):
        parser.error("Unknown target.")
    for command in ("xelatex", "latexmk", "bibtex"):
        if shutil.which(command) is None:
            raise SystemExit(f"Missing {command}. Install TeX Live with XeLaTeX and Chinese language support.")
    env = os.environ.copy()
    if args.tex_support:
        support = args.tex_support.resolve()
        env["TEXINPUTS"] = str(support) + "//:" + env.get("TEXINPUTS", "")
        env["OPENTYPEFONTS"] = str(support) + "//:" + env.get("OPENTYPEFONTS", "")
    for stem in targets:
        cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", f"{stem}.tex"]
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)
        pdf_data = (ROOT / (stem + ".pdf")).read_bytes()
        if not pdf_data.startswith(b"%PDF-") or b"%%EOF" not in pdf_data[-1024:]:
            raise SystemExit(f"Incomplete PDF output: {stem}.pdf. Clean generated TeX state and rebuild.")
        print(f"Built {ROOT / (stem + '.pdf')}", flush=True)

if __name__ == "__main__":
    main()
