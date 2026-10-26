"""The pinned hackathon starter. Its demand functions and thesis prompt text are reused unchanged.

Fetch it with scripts/get_starter.sh; STARTER_DIR points elsewhere if needed. Import fails loudly if the
checkout is missing, at another commit, or has modified tracked files.
"""

import os
import subprocess
import sys
from pathlib import Path

COMMIT = "d585a7b3be50366cc2bdf5789bf8add5890b2abd"
ROOT = Path(os.environ.get("STARTER_DIR", Path(__file__).resolve().parent.parent / "starter"))
PACKAGE = ROOT / "algorithmic-pricing-llms"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)


if not PACKAGE.is_dir():
    raise ImportError(f"starter not found at {ROOT}; run scripts/get_starter.sh")
if (found := _git("rev-parse", "HEAD").stdout.strip()) != COMMIT:
    raise ImportError(f"starter at {ROOT} is at {found or 'unknown'}, expected {COMMIT}")
if _git("diff", "--quiet", "HEAD", "--", "algorithmic-pricing-llms").returncode != 0:
    raise ImportError(f"starter at {ROOT} has modified tracked files")
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from src.environment.pricing_market_logic_multiproduct import get_profits, get_quantities
from src.prompts.prompts import GENERAL_PROMPT, P1, P2

__all__ = ["COMMIT", "GENERAL_PROMPT", "P1", "P2", "get_profits", "get_quantities"]
