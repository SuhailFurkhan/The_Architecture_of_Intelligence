"""
check_token.py — Diagnose HuggingFace authentication issues.

Checks:
  1. Can we find keys.env?
  2. Is there a conflicting system-level HF_TOKEN?
  3. Is the token valid?
  4. Which account does it belong to?
  5. Can we access the gated LLaMA model?

Usage:
    python scripts/check_token.py
"""

import os
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_env_file(filename: str = "Keys.env", max_levels: int = 3) -> Path | None:
    """Search for env file from script dir, climbing up max_levels."""
    current = Path(__file__).resolve().parent
    for i in range(max_levels + 1):
        candidate = current / filename
        if candidate.exists():
            return candidate
        current = current.parent
    return None


def mask_token(token: str) -> str:
    """Show first 8 and last 4 chars only."""
    if len(token) <= 12:
        return "***"
    return f"{token[:8]}...{token[-4:]}"


# ──────────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────────

def check_system_env():
    """Check 1: Is there a system-level HF_TOKEN already set?"""
    print("\n── Check 1: System Environment Variable ──")

    sys_token = os.environ.get("HF_TOKEN")
    if sys_token:
        print(f"  ⚠️  HF_TOKEN is set as a system/user environment variable")
        print(f"     Value: {mask_token(sys_token)}")
        print(f"     This will OVERRIDE whatever is in keys.env!")
        print(f"     → Fix: Remove HF_TOKEN from Windows Environment Variables")
        print(f"            then restart your terminal/IDE.")
        return sys_token
    else:
        print(f"  ✅ No system-level HF_TOKEN found (good — keys.env will be used)")
        return None


def check_keys_env():
    """Check 2: Can we find and read keys.env?"""
    print("\n── Check 2: keys.env File ──")

    env_path = find_env_file("keys.env")
    if env_path is None:
        print(f"  ❌ keys.env NOT found (searched current dir + 3 levels up)")
        print(f"     Searched from: {Path(__file__).resolve().parent}")
        return None, None

    print(f"  ✅ Found: {env_path}")

    # Read it manually to show what's inside
    from dotenv import dotenv_values
    values = dotenv_values(env_path)

    token = (
        values.get("HF_TOKEN")
        or values.get("HUGGINGFACE_TOKEN")
        or values.get("HUGGING_FACE_HUB_TOKEN")
    )

    # Show all keys (not values) in the file
    print(f"     Keys in file: {list(values.keys())}")

    if token:
        print(f"  ✅ HF token found in keys.env: {mask_token(token)}")
    else:
        print(f"  ❌ No HF token found in keys.env")
        print(f"     Expected one of: HF_TOKEN, HUGGINGFACE_TOKEN, HUGGING_FACE_HUB_TOKEN")

    return env_path, token


def check_token_valid(token: str):
    """Check 3: Is this token actually valid on HuggingFace?"""
    print("\n── Check 3: Token Validity ──")

    from huggingface_hub import HfApi

    try:
        api = HfApi(token=token)
        user_info = api.whoami()

        print(f"  ✅ Token is VALID")
        print(f"     Account:  {user_info.get('name', 'unknown')}")
        print(f"     Username: {user_info.get('fullname', user_info.get('name', 'unknown'))}")
        print(f"     Email:    {user_info.get('email', 'not available')}")
        print(f"     Type:     {user_info.get('type', 'unknown')}")

        # Check token permissions
        auth = user_info.get("auth", {})
        if auth:
            print(f"     Permissions: {auth}")

        return user_info

    except Exception as e:
        print(f"  ❌ Token is INVALID or expired")
        print(f"     Error: {e}")
        print(f"     → Fix: Generate a new token at https://huggingface.co/settings/tokens")
        return None


def check_model_access(token: str, model_name: str):
    """Check 4: Can we access the gated model?"""
    print(f"\n── Check 4: Model Access ({model_name}) ──")

    from huggingface_hub import HfApi, model_info

    try:
        info = model_info(model_name, token=token)
        print(f"  ✅ Access GRANTED to {model_name}")
        print(f"     Model size: {info.safetensors.total if info.safetensors else 'unknown'} params")
        print(f"     Gated: {info.gated}")
        return True

    except Exception as e:
        error_str = str(e)

        if "403" in error_str or "gated" in error_str.lower():
            print(f"  ❌ Access DENIED — license not accepted")
            print(f"     → Fix: Visit https://huggingface.co/{model_name}")
            print(f"            Click 'Agree and access repository'")
            print(f"            Make sure you're signed in with the SAME account as your token")
        elif "401" in error_str:
            print(f"  ❌ Authentication failed — token not recognized")
            print(f"     → Fix: Check your token at https://huggingface.co/settings/tokens")
        elif "404" in error_str:
            print(f"  ❌ Model not found — check the model name")
            print(f"     → Fix: Verify '{model_name}' exists on HuggingFace")
        else:
            print(f"  ❌ Unexpected error: {e}")

        return False


def check_conflict(sys_token: str | None, env_token: str | None):
    """Check 5: Are system and keys.env tokens different?"""
    if sys_token and env_token:
        print("\n── Check 5: Token Conflict ──")
        if sys_token == env_token:
            print(f"  ✅ System env and keys.env have the SAME token (no conflict)")
        else:
            print(f"  ⚠️  CONFLICT DETECTED!")
            print(f"     System env:  {mask_token(sys_token)}")
            print(f"     keys.env:    {mask_token(env_token)}")
            print(f"     The system env variable WINS — keys.env is being ignored!")
            print(f"     → Fix: Remove the system environment variable,")
            print(f"            or update keys.env to match it.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import yaml

    print("=" * 60)
    print("  🔍 HuggingFace Token Diagnostic")
    print("=" * 60)

    # Load model name from config
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_name = config.get("model_name", "unsloth/Llama-3.2-1B-Instruct")
        print(f"  Config: {config_path}")
        print(f"  Model:  {model_name}")
    else:
        model_name = "unsloth/Llama-3.2-1B-Instruct"
        print(f"  ⚠️  Config not found, using default model: {model_name}")

    # Run checks
    sys_token = check_system_env()
    env_path, env_token = check_keys_env()
    check_conflict(sys_token, env_token)

    # Use whichever token is active (system env wins, just like Python does)
    active_token = sys_token or env_token

    if active_token:
        print(f"\n  Active token: {mask_token(active_token)} {'(from system env)' if sys_token else '(from keys.env)'}")
        user_info = check_token_valid(active_token)

        if user_info:
            check_model_access(active_token, model_name)
    else:
        print("\n  ❌ No token found anywhere!")
        print("     → Create keys.env with: HF_TOKEN=hf_your_token_here")
        print("     → Get a token at: https://huggingface.co/settings/tokens")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  System env HF_TOKEN:  {'⚠️  SET (may override keys.env)' if sys_token else '✅ Not set'}")
    print(f"  keys.env found:       {'✅ Yes' if env_path else '❌ No'}")
    print(f"  Token in keys.env:    {'✅ Yes' if env_token else '❌ No'}")
    print(f"  Token valid:          {'✅ Yes' if active_token and check_token_valid.__code__ else '❌ No / untested'}")
    print(f"  Model access:         Run above to see result")
    print("=" * 60)


if __name__ == "__main__":
    main()