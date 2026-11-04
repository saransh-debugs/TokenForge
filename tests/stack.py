from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from tests.integration.servers import serve
from tokenforge.serving.gateway.app import create_app as create_gateway
from tokenforge.serving.gateway.settings import GatewaySettings
from tokenforge.serving.mock_engine.app import MockSettings
from tokenforge.serving.mock_engine.app import create_app as create_engine
from tokenforge.serving.mock_engine.engine import EngineConfig

MODEL = "qwen2.5-7b-instruct"


@dataclass
class Stack:
    engine_url: str
    gateway_url: str
    engine_app: Any
    gateway_app: Any

    def faults(self, **kw: Any) -> None:
        httpx.post(f"{self.engine_url}/admin/faults", json=kw).raise_for_status()

    def engine_metric(self, name: str) -> float:
        from tokenforge.serving.gateway.app import parse_metric

        try:
            return parse_metric(httpx.get(f"{self.engine_url}/metrics").text, name)
        except KeyError:
            return 0.0

    def gateway_metric(self, name: str, **labels: str) -> float:
        from prometheus_client.parser import text_string_to_metric_families

        text = httpx.get(f"{self.gateway_url}/metrics").text
        total = 0.0
        for fam in text_string_to_metric_families(text):
            for s in fam.samples:
                if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                    total += s.value
        return total


StackFactory = Callable[..., Stack]


@pytest.fixture
def make_stack() -> Iterator[StackFactory]:
    with ExitStack() as stack:

        def factory(
            engine: EngineConfig | None = None,
            mock: dict[str, Any] | None = None,
            gateway_sndbuf: int | None = None,
            **gateway: Any,
        ) -> Stack:
            eng_cfg = engine or EngineConfig(tpot_base_s=0.002, tpot_per_seq_s=0.0)
            mock_settings = MockSettings(
                model=MODEL, enable_faults=True, engine=eng_cfg, **(mock or {})
            )
            engine_app = create_engine(mock_settings)
            engine_url = stack.enter_context(serve(engine_app))
            gw_settings = dataclasses.replace(
                GatewaySettings(),
                engine_url=engine_url,
                served_model=MODEL,
                engine_model=MODEL,
                poll_interval_s=0.05,
                retry_backoff_s=0.0,
                **gateway,
            )
            gateway_app = create_gateway(gw_settings)
            gateway_url = stack.enter_context(serve(gateway_app, sndbuf=gateway_sndbuf))
            return Stack(engine_url, gateway_url, engine_app, gateway_app)

        yield factory
