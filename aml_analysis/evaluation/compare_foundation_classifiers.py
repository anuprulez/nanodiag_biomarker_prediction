#!/usr/bin/env python3
"""Add exact TabPFN3 and TabICLv2 patient classifiers to both comparison sets."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from post_evaluation_support.foundations import main
if __name__=='__main__':main(ROOT)
