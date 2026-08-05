"""
config/model_config.py - single source of truth for "which Claude model
string does this pipeline use", validated against repro.yaml.

Why this file exists
---------------------
The pipeline log showed:

    [3/6] EnhancedHybridSystemDeFi (core) ...
    ERROR: Error code: 404 - {'type': 'not_found_error',
                               'message': 'model: claude-sonnet-4-20250514'}

A dated model snapshot string got hardcoded somewhere and Anthropic later
retired it. The API returns a 404 *at call time*, deep inside a long run,
instead of failing fast at startup.

repro.yaml already declares the model once, at `llm_model:`. This module
reads that value (no second copy of the string to drift out of sync) and
validates it -- offline against a pinned allow-list, or live via the
`anthropic` SDK -- the same library/exception types `test_key_status.py`
already uses, so behavior is consistent across both scripts.

USAGE
-----
At the top of any pipeline stage, before the first client.messages.create():

    from config.model_config import validate_model, get_configured_model

    model = get_configured_model()   # reads repro.yaml's llm_model
    validate_model(model)            # raises InvalidModelError if bad
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import yaml

# Models known-good as of the time this file was written. Update when
# Anthropic ships new models -- or rely on live validation (see below),
# which doesn't need this list kept manually in sync.
PINNED_VALID_MODELS = {
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
}

# Model strings known to be retired / will 404. Add to this whenever a
# dated snapshot gets found hardcoded somewhere.
KNOWN_RETIRED_MODELS = {
    "claude-sonnet-4-20250514": (
        "This dated snapshot has been retired/renamed. "
        "Use 'claude-sonnet-4-6' instead."
    ),
}

REPRO_YAML_PATH = Path(__file__).parent / "repro.yaml"


class InvalidModelError(ValueError):
    """Raised when a configured model string is known-bad or unverifiable."""


def get_configured_model(repro_yaml_path: Path | str = REPRO_YAML_PATH) -> str:
    """Read llm_model out of repro.yaml -- the single source of truth for
    which model the pipeline should use."""
    with open(repro_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    model = config.get("llm_model")
    if not model:
        raise InvalidModelError(
            f"'llm_model' not found in {repro_yaml_path}. "
            f"repro.yaml must declare llm_model: \"<model-id>\"."
        )
    return model


def validate_model(model: str, *, check_live: bool | None = None) -> None:
    """
    Fail fast if `model` is not a usable Claude model string.

    1. Known-retired?              -> raise with a specific fix.
    2. In the pinned allow-list?   -> pass, no network call.
    3. Otherwise, if an API key is available (or check_live=True), ask the
       live API via the anthropic SDK -- same client/exception types as
       config/test_key_status.py. No key -> raise (fail loudly at startup
       rather than 404 mid-pipeline).

    check_live: None=auto (live-check only if ANTHROPIC_API_KEY is set and
    model isn't pinned), True=always hit the API, False=never hit the API.
    """
    if model in KNOWN_RETIRED_MODELS:
        raise InvalidModelError(
            f"Model '{model}' is retired. {KNOWN_RETIRED_MODELS[model]}"
        )

    if model in PINNED_VALID_MODELS:
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    should_check_live = check_live if check_live is not None else bool(api_key)

    if not should_check_live:
        raise InvalidModelError(
            f"Model '{model}' is not in the pinned allow-list "
            f"({sorted(PINNED_VALID_MODELS)}) and no ANTHROPIC_API_KEY was "
            f"found to verify it live. Refusing to proceed."
        )

    if not api_key:
        raise InvalidModelError(
            f"Model '{model}' needs live verification but ANTHROPIC_API_KEY "
            f"is not set."
        )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        # Cheapest possible live check: a 1-token round-trip, same pattern
        # as config/test_key_status.py's validation call.
        client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}],
        )
    except anthropic.NotFoundError as e:
        raise InvalidModelError(f"Model '{model}' does not exist (404): {e}") from e
    except anthropic.AuthenticationError as e:
        raise InvalidModelError(f"Cannot verify model '{model}': API key invalid: {e}") from e
    # Rate limits / overloaded / transient errors are NOT model problems --
    # let those propagate as-is rather than mislabeling them InvalidModelError.
