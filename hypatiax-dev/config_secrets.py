# ── API Key Setup — Kaggle & Colab ──────────────────────────────────────────
import os

def _load_api_key():
    # 1. Already set in environment — nothing to do
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✅ ANTHROPIC_API_KEY already set")
        return

    # 2. Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = key
        print("✅ ANTHROPIC_API_KEY loaded from Kaggle secrets")
        return
    except Exception:
        pass

    # 3. Local .env file at repo root
    try:
        from dotenv import load_dotenv
        repo_root = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(repo_root, ".env")
        load_dotenv(dotenv_path)
        if os.getenv("ANTHROPIC_API_KEY"):
            print("✅ ANTHROPIC_API_KEY loaded from .env")
            return
    except Exception:
        pass

    # 4. Google Colab
    try:
        from google.colab import userdata
        key = userdata.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = key
        print("✅ ANTHROPIC_API_KEY loaded from Colab secrets")
        return
    except Exception:
        pass

    print("⚠️  ANTHROPIC_API_KEY not found — LLM mode will be disabled. "
          "Set it in Kaggle/Colab secrets, a local .env file, or via os.environ.")

_load_api_key()
