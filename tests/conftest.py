import os
import sys
from pathlib import Path

os.environ.setdefault("ARBITER_DEBUG", "1")  # KB.check() after every apply (E3)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
