"""
scripts/tests/test_model_validation.py

CI regression test for the 404 seen in the pipeline log:

    ERROR: Error code: 404 - {'type': 'not_found_error',
                               'message': 'model: claude-sonnet-4-20250514'}

Run locally / in CI with:

    pytest scripts/tests/test_model_validation.py -v

Tier 1 (always runs, offline): validates against the pinned allow-list and
the model declared in config/repro.yaml.
Tier 2 (runs only if ANTHROPIC_API_KEY is set, e.g. nightly schedule):
cross-checks the pinned list and repro.yaml's model against the live API.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# config/ is a sibling of scripts/ at repo root -- add repo root to path so
# `from config.model_config import ...` resolves the same way pipeline code
# importing this module would.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config.model_config import (  # noqa: E402
    validate_model,
    get_configured_model,
    InvalidModelError,
    PINNED_VALID_MODELS,
    KNOWN_RETIRED_MODELS,
)


# ---------------------------------------------------------------------------
# Tier 1: offline, runs on every commit
# ---------------------------------------------------------------------------

def test_known_retired_model_is_rejected():
    """Reproduces the exact failure from the log."""
    with pytest.raises(InvalidModelError, match="retired"):
        validate_model("claude-sonnet-4-20250514", check_live=False)


@pytest.mark.parametrize("model", sorted(PINNED_VALID_MODELS))
def test_pinned_models_pass_without_network(model):
    validate_model(model, check_live=False)


def test_unknown_model_without_network_is_rejected():
    with pytest.raises(InvalidModelError):
        validate_model("totally-made-up-model-string", check_live=False)


def test_pinned_and_retired_lists_dont_overlap():
    assert PINNED_VALID_MODELS.isdisjoint(KNOWN_RETIRED_MODELS)


def test_repro_yaml_declares_a_model():
    """config/repro.yaml must have a non-empty llm_model key."""
    model = get_configured_model()
    assert model and isinstance(model, str)


def test_repro_yaml_model_is_not_retired():
    """The model repro.yaml currently points to (llm_model: claude-sonnet-4-6
    as of this writing) must not be on the known-retired list. This is the
    test that would have caught the log's failure if repro.yaml had still
    pointed at claude-sonnet-4-20250514."""
    model = get_configured_model()
    validate_model(model, check_live=False)


def test_pipeline_call_site_validates_before_request():
    """Simulates the fix: validate BEFORE calling the API."""
    bad_model = "claude-sonnet-4-20250514"

    def run_pipeline_stage(model: str):
        validate_model(model, check_live=False)  # <-- call this first
        raise AssertionError("should never reach the API call")

    with pytest.raises(InvalidModelError, match="retired"):
        run_pipeline_stage(bad_model)


# ---------------------------------------------------------------------------
# Tier 2: live, only runs with a key present
# ---------------------------------------------------------------------------

requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live API checks",
)


@requires_api_key
@pytest.mark.parametrize("model", sorted(PINNED_VALID_MODELS))
def test_pinned_models_are_actually_live(model):
    """Guards against the pinned allow-list itself going stale."""
    validate_model(model, check_live=True)


@requires_api_key
def test_repro_yaml_model_is_actually_live():
    """The model repro.yaml currently points to must actually exist."""
    model = get_configured_model()
    validate_model(model, check_live=True)


@requires_api_key
def test_known_retired_model_is_actually_gone_live():
    with pytest.raises(InvalidModelError):
        validate_model("claude-sonnet-4-20250514", check_live=True)
