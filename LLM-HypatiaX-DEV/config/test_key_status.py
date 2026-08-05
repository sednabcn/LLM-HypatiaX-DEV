"""
test_key_status.py - validate ANTHROPIC_API_KEY with a live round-trip.

LOCAL USE:
    Set ANTHROPIC_API_KEY in your shell or hypatiax/.env, then:
        python3 config/test_key_status.py

CI USE (GitHub Actions):
    The key is injected via the workflow's `env:` block:
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    load_dotenv() is called first so local runs still work; in CI the
    secret is already in the environment and load_dotenv() is a no-op.

Exit codes:
    0  key is active and working
    1  key missing, invalid, revoked, or unexpected error
    2  key active but account has no credits (run will fail later anyway)
"""

import os
import sys
import time
import anthropic

# --- retry config ---
MAX_RETRIES = 5
BASE_DELAY = 2  # seconds


# load_dotenv() for local runs only; in CI ANTHROPIC_API_KEY is already set
# via the workflow env block and load_dotenv() is a no-op.
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads hypatiax/.env if present; ignored in CI
except ImportError:
    pass  # python-dotenv not installed in this env - fine for CI

api_key = os.environ.get("ANTHROPIC_API_KEY", "")

if not api_key:
    print("❌ ANTHROPIC_API_KEY is not set or empty.")
    print("   In CI: ensure the secret is mapped via `env: ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}`")
    print("   Locally: set it in your shell or in hypatiax/.env")
    sys.exit(1)

print(f"ANTHROPIC_API_KEY present: YES ({len(api_key)} chars, prefix={api_key[:7]}...)")

client = anthropic.Anthropic(api_key=api_key)

for attempt in range(1, MAX_RETRIES + 1):
    try:
        print(f"\nValidation attempt {attempt}/{MAX_RETRIES}...")

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}],
        )

        print("✅ Key is ACTIVE and WORKING")
        print(f"   Response: {response.content[0].text!r}")
        sys.exit(0)

    except anthropic.AuthenticationError as e:
        print("❌ Key is INVALID or REVOKED")
        print(f"   Error: {e}")
        sys.exit(1)

    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower():
            print("⚠️  Key is ACTIVE but account has NO CREDITS")
            print(f"   Error: {e}")
            print(
                "   The experiment run will fail when it tries to call the API."
            )
            sys.exit(2)

        print(f"⚠️  BadRequestError (unexpected): {e}")
        sys.exit(1)

    # ---- transient/retryable failures ----
    except anthropic.APIError as e:
        if getattr(e, "status_code", None) == 529:
            # handle overloaded server
            print("⚠️ Anthropic servers overloaded (HTTP 529)")
            print("   Recommendation: wait a few seconds and retry; consider a lighter model.")
            first_error = False
        else:
            # other API errors
            print("❌ Unexpected API error")
            sys.exit(1)

    except anthropic.RateLimitError as e:
        print(f"⚠️  Rate limited: {e}")

    except (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    ) as e:
        print(f"⚠️  Temporary API failure: {type(e).__name__}: {e}")

    # ---- unknown fatal ----

    except Exception as e:
        print(
            f"❌ Unexpected error during key validation: "
            f"{type(e).__name__}: {e}"
        )
        sys.exit(1)

    # retry if attempts remain
    if attempt < MAX_RETRIES:
        delay = BASE_DELAY * attempt
        print(f"   Retrying in {delay}s...")
        time.sleep(delay)

print("\n❌ Validation failed after maximum retries")
sys.exit(1)
