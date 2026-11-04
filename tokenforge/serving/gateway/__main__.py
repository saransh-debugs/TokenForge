"""``tokenforge-gateway``: run the gateway under uvicorn."""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    from tokenforge.serving.gateway.app import create_app
    from tokenforge.serving.gateway.observability import configure_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    configure_logging(args.log_level)
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
