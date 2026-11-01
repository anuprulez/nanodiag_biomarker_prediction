#!/usr/bin/env python3
"""Follow up the fixed five-run PNA candidate panel at the patient level."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from post_evaluation_support.analysis import main

if __name__ == '__main__':
    main(ROOT)
