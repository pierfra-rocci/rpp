from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web.cli import main as streamlit_main


def main() -> int:
    frontend_path = Path(__file__).with_name("frontend.py")
    sys.argv = ["streamlit", "run", str(frontend_path)]
    return streamlit_main()


if __name__ == "__main__":
    raise SystemExit(main())
