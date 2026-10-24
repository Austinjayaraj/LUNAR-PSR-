# Forwarding wrapper for ml/training/train.py
import runpy, sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "ml/training/train.py"), run_name="__main__")
