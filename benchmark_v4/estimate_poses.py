#!/usr/bin/env python3
"""Delegate to the exported MCV estimator with unchanged numerical settings."""
import runpy
if __name__=="__main__":
    runpy.run_module("controller.multiworldbench_v3.tools.estimate_navigation_mcv18",run_name="__main__")
