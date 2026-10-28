"""Start the configured end-to-end research campaign."""

from __future__ import annotations

from .configuration import load_config
from .training_handoff import frozen_training_config


def run_configured():
    """Run the campaign using the project configuration."""
    settings = frozen_training_config(load_config())
    if settings["application"]["mode"] != "research_campaign":
        raise ValueError("application.mode must be research_campaign")
    from .research_campaign import run_research_campaign

    return run_research_campaign(settings)
