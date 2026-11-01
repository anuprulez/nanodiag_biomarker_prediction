#!/usr/bin/env python3
"""NIAPU-only rankings, external matching and patient classifiers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from post_evaluation_support.niapu_rankings import main

if __name__=='__main__':main(ROOT)
