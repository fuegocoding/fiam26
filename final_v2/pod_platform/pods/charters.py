"""The four research pods, one per ``strategies.base.Charter``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from ..strategies.base import Charter


@dataclass(frozen=True)
class PodSpec:
    pod_id: str
    charter: Charter
    namespace: str
    mission: str
    mechanism_families: tuple[str, ...]


POD_SPECS: dict[Charter, PodSpec] = {
    Charter.FUNDAMENTALS_EXPECTATIONS: PodSpec(
        pod_id="fundamentals",
        charter=Charter.FUNDAMENTALS_EXPECTATIONS,
        namespace="pod_fundamentals",
        mission="Forecast business fundamentals and analyst-expectation revisions before they are priced.",
        mechanism_families=("earnings_surprise", "estimate_revision", "profitability_quality"),
    ),
    Charter.CORPORATE_EVENTS: PodSpec(
        pod_id="events",
        charter=Charter.CORPORATE_EVENTS,
        namespace="pod_events",
        mission="Study corporate events and capital-allocation decisions and their delayed price response.",
        mechanism_families=("net_issuance", "buyback_dividend", "investment_and_acquisition"),
    ),
    Charter.VALUATION_RELATIVE_VALUE: PodSpec(
        pod_id="valuation",
        charter=Charter.VALUATION_RELATIVE_VALUE,
        namespace="pod_valuation",
        mission="Compare price with fundamental value and peers to find durable relative mispricing.",
        mechanism_families=("value_spread", "peer_relative", "intrinsic_value"),
    ),
    Charter.MARKET_DYNAMICS: PodSpec(
        pod_id="dynamics",
        charter=Charter.MARKET_DYNAMICS,
        namespace="pod_dynamics",
        mission="Model market dynamics and information diffusion: momentum, flows, short interest, liquidity.",
        mechanism_families=("momentum", "information_diffusion", "short_interest_liquidity"),
    ),
}

_BY_KEY: dict[str, PodSpec] = {}
for _spec in POD_SPECS.values():
    for _key in (_spec.pod_id, _spec.namespace, _spec.charter.value):
        _BY_KEY[_key] = _spec


def pod_spec(key: Union[Charter, str]) -> PodSpec:
    """Resolve a Charter, charter value, pod id or namespace."""
    if isinstance(key, Charter):
        return POD_SPECS[key]
    try:
        return _BY_KEY[str(key)]
    except KeyError:
        raise KeyError(f"unknown pod {key!r}; known: {sorted(s.pod_id for s in POD_SPECS.values())}") from None
