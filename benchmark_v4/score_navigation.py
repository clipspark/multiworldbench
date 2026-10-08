#!/usr/bin/env python3
"""Portable V4 MCV scoring entry; preserves the recorded V4 numerical protocol."""
import argparse
from pathlib import Path
from controller.multiworldbench_v3.tools.score_navigation90_mcv import main, MODELS
if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--estimates-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--models",nargs="+",choices=MODELS,default=list(MODELS))
    p.add_argument("--pose-gap-seconds",type=float,default=0.,help="Optional diagnostic only; released baseline uses 0")
    a=p.parse_args()
    if a.pose_gap_seconds<0:p.error("--pose-gap-seconds must be nonnegative")
    main(a.run,apply_calibration=False,estimator="mcv",consistency_threshold=.60,
         output_dir=a.output_dir,estimates_root=a.estimates_root,
         selected_models=a.models,pose_gap_seconds=a.pose_gap_seconds)
