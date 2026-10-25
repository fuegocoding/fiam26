import datetime as dt

import pytest

from pod_platform.agents.registry import ModelRegistry, TemporalPolicy
from pod_platform.contracts.errors import PodPlatformError
from pod_platform.orchestration.freeze import freeze_run
from .test_agents import T0, _adoption, _checkpoint
from .test_freeze_submission import _freeze_kwargs, _manifest, _outputs


def test_future_adoption_is_not_used_in_an_earlier_replay():
    registry = ModelRegistry()
    registry.register(_checkpoint())
    registry.adopt(_adoption(effective_start=T0 + dt.timedelta(days=1)))
    with pytest.raises(PodPlatformError):
        registry.route("lead", T0)
    assert registry.route("lead", T0 + dt.timedelta(days=1)) == "ckpt_1"


def test_policy_b_adoption_cannot_leak_into_release_gated_route():
    registry = ModelRegistry(allow_cutoff_qualified_backcast=True)
    registry.register(_checkpoint())
    registry.adopt(_adoption(temporal_policy=TemporalPolicy.CUTOFF_QUALIFIED_BACKCAST,
                            eligibility_certificate_id="cert_b"))
    with pytest.raises(PodPlatformError):
        registry.route("lead", T0, policy=TemporalPolicy.RELEASE_GATED)


def test_material_llm_intervention_requires_actual_counterfactual():
    with pytest.raises(PodPlatformError):
        freeze_run(_manifest(), _outputs(),
                   **_freeze_kwargs(material_agent_interventions=1,
                                    agent_counterfactual_present=False))
    result = freeze_run(_manifest(), _outputs(),
                        **_freeze_kwargs(material_agent_interventions=1,
                                         agent_counterfactual_present=True))
    assert result.accepted
