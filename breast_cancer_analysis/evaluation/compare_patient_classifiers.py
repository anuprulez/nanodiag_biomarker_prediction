#!/usr/bin/env python3
"""Compare patient classifiers using validated cached post-evaluation inputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from post_evaluation_support.classifiers import main

if __name__=='__main__':main(ROOT)
