import os
import sys
from pathlib import Path

import tempfile

os.environ.setdefault("ARBITER_DEBUG", "1")  # KB.check() after every apply (E3)
os.environ["ARBITER_TEST_LOG"] = str(Path(tempfile.mkdtemp()) / "test_reads.log")  # never touch the repo ledger
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
