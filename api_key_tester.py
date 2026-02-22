"""
API Key Diagnostic Script
=========================
Run from the project root:  python check_api_keys.py

Checks:
  1. Keys.env file — exists, readable, correct format
  2. Key value — length, prefix, whitespace issues
  3. Live validation — actual API call to confirm the key works
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✅  {msg}{RESET}")
def fail(msg):  print(f"  {RED}❌  {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠️   {msg}{RESET}")
def info(msg):  print(f"  {CYAN}ℹ️   {msg}{RESET}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}\n{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Locate Keys.env
# ─────────────────────────────────────────────────────────────────────────────

def find_env_file() -> Path | None:
    search_paths = [
        Path.cwd() / "Keys.env",
        Path.cwd() / ".env",
        Path(__file__).parent / "Keys.env",
        Path(__file__).parent / ".env",
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None


def check_env_file(env_path: Path) -> dict:
    """Parse Keys.env and return raw key-value pairs."""
    raw_vars = {}
    issues   = []

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        issues.append(f"Cannot read file: {e}")
        return {"vars": raw_vars, "issues": issues}

    for i, line in enumerate(lines, 1):
        original = line
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            issues.append(f"Line {i}: no '=' found → '{original}'")
            continue

        key, value = line.split("=", 1)

        # Detect space around =
        if key != key.strip() or value != value.strip():
            issues.append(
                f"Line {i}: whitespace around '=' "
                f"→ '{original.rstrip()}'"
            )

        key   = key.strip()
        value = value.strip()

        # Strip surrounding quotes
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]

        # Detect trailing whitespace (after quote stripping)
        if value != value.strip():
            issues.append(
                f"Line {i}: trailing whitespace in value for '{key}'"
            )
            value = value.strip()

        raw_vars[key] = value

    return {"vars": raw_vars, "issues": issues}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Validate key format
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED = {
    "ANTHROPIC_API_KEY": {
        "prefix":  "sk-ant-api03-",
        "length":  108,
        "env_var": "ANTHROPIC_API_KEY",
    },
    "HF_TOKEN": {
        "prefix":  "hf_",
        "length":  None,   # variable length
        "env_var": "HF_TOKEN",
    },
    "OPENAI_API_KEY": {
        "prefix":  "sk-",
        "length":  None,
        "env_var": "OPENAI_API_KEY",
    },
}


def validate_key_format(key_name: str, value: str):
    spec = EXPECTED.get(key_name)
    if not spec:
        info(f"{key_name}: unknown key type, skipping format check")
        return

    # Prefix check
    if not value.startswith(spec["prefix"]):
        fail(
            f"{key_name}: wrong prefix — "
            f"expected '{spec['prefix']}...', "
            f"got '{value[:len(spec['prefix'])+4]}...'"
        )
    else:
        ok(f"{key_name}: prefix '{spec['prefix']}' ✓")

    # Length check
    if spec["length"]:
        if len(value) != spec["length"]:
            fail(
                f"{key_name}: wrong length — "
                f"expected {spec['length']} chars, got {len(value)}"
            )
        else:
            ok(f"{key_name}: length {len(value)} chars ✓")
    else:
        info(f"{key_name}: length {len(value)} chars (variable length key)")

    # Whitespace
    if value != value.strip():
        fail(f"{key_name}: contains leading/trailing whitespace")
    else:
        ok(f"{key_name}: no whitespace issues ✓")

    # Newline chars
    if "\n" in value or "\r" in value:
        fail(f"{key_name}: contains newline character — key is corrupted")
    else:
        ok(f"{key_name}: no newline chars ✓")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Live API validation
# ─────────────────────────────────────────────────────────────────────────────

def live_check_anthropic(api_key: str) -> bool:
    try:
        import anthropic
    except ImportError:
        warn("anthropic package not installed — skipping live check")
        warn("Install with:  pip install anthropic")
        return False

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 10,
            messages   = [{"role": "user", "content": "Say OK"}],
        )
        ok(f"Anthropic live call succeeded — model: {response.model}")
        return True
    except anthropic.AuthenticationError:
        fail("Anthropic: 401 Authentication error — key is invalid or revoked")
        info("Generate a new key at: https://console.anthropic.com")
        return False
    except anthropic.PermissionDeniedError:
        fail("Anthropic: 403 Permission denied — key may lack required permissions")
        return False
    except anthropic.RateLimitError:
        warn("Anthropic: Rate limit hit — but the key IS valid")
        return True
    except Exception as e:
        fail(f"Anthropic: unexpected error — {e}")
        return False


def live_check_hf(token: str) -> bool:
    try:
        import requests
    except ImportError:
        warn("requests package not installed — skipping HF live check")
        return False

    try:
        r = requests.get(
            "https://huggingface.co/api/whoami",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if r.status_code == 200:
            user = r.json().get("name", "unknown")
            ok(f"HuggingFace token valid — logged in as: {user}")
            return True
        elif r.status_code == 401:
            fail("HuggingFace: 401 — token is invalid or expired")
            info("Generate a new token at: https://huggingface.co/settings/tokens")
            return False
        else:
            warn(f"HuggingFace: unexpected status {r.status_code}")
            return False
    except Exception as e:
        fail(f"HuggingFace: request failed — {e}")
        return False


def live_check_openai(api_key: str) -> bool:
    try:
        import openai
    except ImportError:
        warn("openai package not installed — skipping live check")
        return False

    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        ok("OpenAI live call succeeded")
        return True
    except openai.AuthenticationError:
        fail("OpenAI: 401 Authentication error — key is invalid or revoked")
        info("Generate a new key at: https://platform.openai.com/api-keys")
        return False
    except Exception as e:
        fail(f"OpenAI: unexpected error — {e}")
        return False


LIVE_CHECKERS = {
    "ANTHROPIC_API_KEY": live_check_anthropic,
    "HF_TOKEN":          live_check_hf,
    "OPENAI_API_KEY":    live_check_openai,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*55}")
    print("  API Key Diagnostic")
    print(f"{'='*55}{RESET}")

    # ── 1. Find Keys.env ──────────────────────────────────────
    header("STEP 1 — Locate Keys.env")

    env_path = find_env_file()
    if env_path is None:
        fail("Keys.env not found in any search path")
        info("Search order:")
        info("  ./Keys.env")
        info("  ./.env")
        info("  <script dir>/Keys.env")
        info("  <script dir>/.env")
        print()
        sys.exit(1)

    ok(f"Found: {env_path}")

    result = check_env_file(env_path)
    parsed = result["vars"]
    issues = result["issues"]

    if issues:
        for issue in issues:
            fail(f"Format issue: {issue}")
    else:
        ok("File format looks clean")

    if not parsed:
        fail("No key=value pairs found in file")
        sys.exit(1)

    info(f"Keys found in file: {list(parsed.keys())}")

    # Also check system environment for any keys not in the file
    for key_name in EXPECTED:
        if key_name not in parsed and os.environ.get(key_name):
            warn(f"{key_name} not in Keys.env but IS set in system environment")
            parsed[key_name] = os.environ[key_name]

    # ── 2. Format validation ──────────────────────────────────
    header("STEP 2 — Key Format Validation")

    keys_to_check = {k: v for k, v in parsed.items() if k in EXPECTED}

    if not keys_to_check:
        warn("No recognised API keys found (ANTHROPIC_API_KEY / HF_TOKEN / OPENAI_API_KEY)")
    else:
        for key_name, value in keys_to_check.items():
            preview = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"\n  [{key_name}]  ({preview})")
            validate_key_format(key_name, value)

    # ── 3. Live API checks ────────────────────────────────────
    header("STEP 3 — Live API Validation")

    if not keys_to_check:
        warn("No keys to validate")
    else:
        for key_name, value in keys_to_check.items():
            checker = LIVE_CHECKERS.get(key_name)
            if checker:
                print(f"\n  [{key_name}]")
                checker(value)
            else:
                info(f"No live checker for {key_name}")

    # ── Summary ───────────────────────────────────────────────
    header("SUMMARY")

    if not keys_to_check:
        fail("No API keys found — add them to Keys.env")
        info("Example Keys.env contents:")
        print()
        print("    ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxx")
        print("    HF_TOKEN=hf_xxxxxxxxxxxx")
        print("    OPENAI_API_KEY=sk-xxxxxxxxxxxx")
    else:
        for key_name in keys_to_check:
            info(f"{key_name} was checked — see results above")

    print()


if __name__ == "__main__":
    main()