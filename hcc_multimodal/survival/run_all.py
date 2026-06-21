"""Orchestrator — run the full survival pipeline (Tasks 1 -> 2 -> 3).

Run:  python -m hcc_multimodal.survival.run_all
"""

from __future__ import annotations

import argparse

from . import compute_risk_scores, make_figures, run_analysis, screen, screen_figures


def main() -> None:
    print("=== Task 1: compute risk scores ===")
    compute_risk_scores.main(argparse.Namespace(n_components=10))
    print("\n=== Task 2: survival analysis ===")
    run_analysis.main(argparse.Namespace())
    print("\n=== Task 3: figures ===")
    make_figures.main(argparse.Namespace(report_dir=make_figures.DEFAULT_REPORT_DIR))
    print("\n=== Task 4: model screen (all models + radiomic baseline) ===")
    screen.main(argparse.Namespace())
    print("\n=== Task 5: screen scatter figure ===")
    screen_figures.main(argparse.Namespace(out=screen_figures.DEFAULT_OUT))


if __name__ == "__main__":
    main()
