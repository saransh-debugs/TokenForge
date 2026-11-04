"""Smoke test a running stack: readiness, one completion, one stream ending in [DONE]."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MODEL = "qwen2.5-7b-instruct"


def _post(url: str, body: dict[str, object], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            with urllib.request.urlopen(f"{args.url}/health/ready", timeout=2) as r:  # noqa: S310
                if r.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            print("smoke: gateway never became ready", file=sys.stderr)
            return 1
        time.sleep(1)

    status, body = _post(f"{args.url}/v1/completions",
                         {"model": MODEL, "prompt": "smoke test", "max_tokens": 4}, 30)
    if status != 200 or json.loads(body)["usage"]["completion_tokens"] != 4:
        print(f"smoke: completion failed: {status} {body[:200]!r}", file=sys.stderr)
        return 1
    status, body = _post(f"{args.url}/v1/completions",
                         {"model": MODEL, "prompt": "smoke", "max_tokens": 4, "stream": True}, 30)
    events = [ln for ln in body.decode().splitlines() if ln.startswith("data: ")]
    if status != 200 or events[-1] != "data: [DONE]":
        print(f"smoke: stream failed: {status} {events[-2:]}", file=sys.stderr)
        return 1
    print(f"smoke: ok ({len(events) - 1} stream events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
