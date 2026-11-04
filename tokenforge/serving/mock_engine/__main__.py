"""``tokenforge-mock-engine``: run the mock engine under uvicorn."""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    from tokenforge.serving.mock_engine.app import create_app

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
