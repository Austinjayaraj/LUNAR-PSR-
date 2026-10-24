# Forwarding wrapper for scripts/check_deps.py
import runpy, sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(ROOT, "scripts/check_deps.py"), run_name="__main__")
