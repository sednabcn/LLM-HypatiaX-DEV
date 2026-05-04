import os

import anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

try:
    client = anthropic.Anthropic(api_key=api_key)
    # Try to make a minimal request
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}],
    )
    print("✅ Key is ACTIVE and WORKING")
    print(f"Response: {response.content[0].text}")
except anthropic.AuthenticationError as e:
    print("❌ Key is INVALID or REVOKED")
    print(f"Error: {e}")
except anthropic.BadRequestError as e:
    if "credit balance" in str(e).lower():
        print("✅ Key is ACTIVE but NO CREDITS")
        print(f"Error: {e}")
    else:
        print("⚠️ Other error:", e)
except Exception as e:
    print("⚠️ Unexpected error:", e)
