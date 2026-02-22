"""
additive_main.py — Master controller for Additive PEFT (Bottleneck & (IA)³).

Mirrors the structure of peft_main.py exactly, with additive-specific steps.

Usage:
    python additive_main.py                     # Interactive menu
    python additive_main.py --run all           # Full pipeline
    python additive_main.py --run train         # Just train
    python additive_main.py --run vram          # Just check VRAM
    python additive_main.py --run compare       # Analyze what adapter learned
    python additive_main.py --prompt "..."      # Custom inference prompt
    python additive_main.py --method ia3        # Switch to (IA)³ (overrides config)
    python additive_main.py --run all --yes     # Auto-confirm all prompts
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "additive_training_config.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Model discovery helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_adapter_dir(path: Path) -> bool:
    """
    Return True if `path` looks like a saved adapter or merged model output.

    Recognises three layouts:
      1. HuggingFace PEFT adapter:    path/adapter_config.json
      2. AdapterHub bottleneck style: path/task_adapter/adapter_config.json
      3. Merged (IA)³ model:          path/config.json  (no adapter_config)
    """
    return (
        (path / "adapter_config.json").exists()
        or (path / "task_adapter" / "adapter_config.json").exists()
        or (path / "config.json").exists()
    )


def _read_adapter_method(path: Path) -> str:
    """Return the adapter method string stored in the config, or a best guess."""
    import json

    # PEFT layout
    cfg_file = path / "adapter_config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                cfg = json.load(f)
            peft_type = cfg.get("peft_type", "").lower()
            if "ia3" in peft_type:
                return "ia3"
            if "bottleneck" in peft_type or "adapter" in peft_type:
                return "bottleneck"
            if "lora" in peft_type:
                return "lora"
        except Exception:
            pass

    # AdapterHub layout
    task_cfg = path / "task_adapter" / "adapter_config.json"
    if task_cfg.exists():
        return "bottleneck"

    # Merged model (no adapter_config)
    if (path / "config.json").exists():
        return "merged (ia3)"

    return "unknown"


def _dir_size_mb(path: Path) -> float:
    """Sum file sizes under `path` in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _model_details(path: Path) -> dict:
    """
    Collect human-readable details about a saved model directory.

    Returns a dict with keys:
      path, method, size_mb, modified, base_model, trainable_params
    """
    import json
    import datetime

    method = _read_adapter_method(path)
    size_mb = _dir_size_mb(path)

    # Modification time — use the most recently touched file
    all_files = list(path.rglob("*"))
    if all_files:
        latest_mtime = max(f.stat().st_mtime for f in all_files if f.is_file())
        modified = datetime.datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M")
    else:
        modified = "unknown"

    # Base model name
    base_model = "unknown"
    for cfg_candidate in [
        path / "adapter_config.json",
        path / "task_adapter" / "adapter_config.json",
        path / "config.json",
    ]:
        if cfg_candidate.exists():
            try:
                with open(cfg_candidate) as f:
                    cfg = json.load(f)
                base_model = (
                    cfg.get("base_model_name_or_path")
                    or cfg.get("_name_or_path")
                    or "unknown"
                )
                if base_model != "unknown":
                    break
            except Exception:
                pass

    # Trainable param count (adapter weights only) — rough estimate from file size
    adapter_bin = path / "adapter_model.safetensors"
    if not adapter_bin.exists():
        adapter_bin = path / "adapter_model.bin"
    params_str = "unknown"
    if adapter_bin.exists():
        kb = adapter_bin.stat().st_size / 1024
        params_str = f"~{kb:.0f} KB adapter weights"

    return {
        "path": path,
        "method": method,
        "size_mb": size_mb,
        "modified": modified,
        "base_model": base_model,
        "adapter_weights": params_str,
    }


def find_trained_models(output_dir: str) -> list[dict]:
    """
    Search for previously trained models in three directory scopes:

      1. Current working directory    (cwd)
      2. Parent of cwd                (cwd/..)
      3. Grandparent of cwd           (cwd/../..)

    In each scope we look for:
      a) The exact output_dir path (the canonical save location)
      b) Any sub-directory named outputs/llama-additive-*
      c) Any sub-directory that looks like an adapter save (adapter_config.json, etc.)

    Returns a deduplicated list of detail dicts sorted by modification time
    (most recent first).
    """
    search_roots = []
    cwd = Path.cwd()
    for level in range(3):          # 0 = cwd, 1 = parent, 2 = grandparent
        root = cwd
        for _ in range(level):
            root = root.parent
        search_roots.append(root)

    canonical = Path(output_dir).resolve()
    found: dict[Path, dict] = {}   # resolved_path -> details, for dedup

    def _register(p: Path):
        rp = p.resolve()
        if rp not in found and _is_adapter_dir(p):
            found[rp] = _model_details(p)

    # Always check the canonical output path first
    if _is_adapter_dir(canonical):
        _register(canonical)

    for root in search_roots:
        # Check root itself
        if _is_adapter_dir(root):
            _register(root)

        # Check one level of sub-dirs
        if root.is_dir():
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                if _is_adapter_dir(child):
                    _register(child)
                # Also check outputs/ sub-dirs (common pattern)
                if child.name == "outputs":
                    for grandchild in child.iterdir():
                        if grandchild.is_dir() and _is_adapter_dir(grandchild):
                            _register(grandchild)

    # Sort by modification time, most recent first
    sorted_models = sorted(
        found.values(),
        key=lambda d: d["modified"],
        reverse=True,
    )
    return sorted_models


def print_model_details(details: dict, index: int = None):
    """Pretty-print the details of a discovered model."""
    prefix = f"  [{index}] " if index is not None else "  "
    print(f"\n{prefix}📦  {details['path']}")
    print(f"      Method:         {details['method'].upper()}")
    print(f"      Base model:     {details['base_model']}")
    print(f"      Adapter size:   {details['adapter_weights']}")
    print(f"      Total on disk:  {details['size_mb']:.1f} MB")
    print(f"      Last modified:  {details['modified']}")


def prompt_use_or_retrain(
    found_models: list[dict],
    output_dir: str,
    auto_yes: bool = False,
) -> str:
    """
    Show discovered models and ask the user what to do.

    Returns one of:
      "use_existing"  — caller should skip training and use the found model
      "retrain"       — caller should proceed with training (old model will be overwritten)
      "cancelled"     — user chose to abort
    """
    print("\n" + "─" * 60)
    print("  🔍  Previously trained model(s) found:")
    print("─" * 60)

    for i, details in enumerate(found_models):
        print_model_details(details, index=i + 1)

    print("\n" + "─" * 60)
    print("  What would you like to do?\n")
    print("    [U]  Use an existing model  (skip training)")
    print("    [T]  Train a new model      (⚠️  will OVERWRITE the model at")
    print(f"                                   {output_dir})")
    print("    [C]  Cancel / go back")
    print()

    if auto_yes:
        print("  [auto-yes] Proceeding with training (overwrite existing model).")
        return "retrain"

    while True:
        choice = input("  Enter choice [U/T/C]: ").strip().lower()
        if choice in ("u", "use"):
            # If multiple models found, ask which one
            if len(found_models) == 1:
                selected = found_models[0]
            else:
                print("\n  Which model would you like to use?")
                for i, d in enumerate(found_models):
                    print(f"    [{i+1}]  {d['path']}")
                while True:
                    try:
                        n = int(input("  Enter number: ").strip())
                        if 1 <= n <= len(found_models):
                            selected = found_models[n - 1]
                            break
                    except ValueError:
                        pass
                    print(f"  Please enter a number between 1 and {len(found_models)}.")

            print(f"\n  ✅ Using existing model at: {selected['path']}")
            # Patch the canonical output_dir in the caller by storing path on dict
            selected["_chosen"] = True
            return "use_existing", selected["path"]

        elif choice in ("t", "train"):
            # Extra confirmation before overwriting
            target = Path(output_dir)
            if target.exists():
                print(f"\n  ⚠️  WARNING: Training will DELETE and OVERWRITE:")
                print(f"       {target.resolve()}")
                confirm_input = input("\n  Type YES to confirm overwrite: ").strip()
                if confirm_input != "YES":
                    print("  Overwrite cancelled.  Returning to menu.")
                    return "cancelled", None
            print(f"\n  Proceeding with training.  Output → {output_dir}")
            return "retrain", None

        elif choice in ("c", "cancel", "q"):
            return "cancelled", None
        else:
            print("  Invalid choice — enter U, T, or C.")


def run_step(script: str, extra_args: list[str] = None, desc: str = ""):
    """Run a pipeline step as a subprocess."""
    cmd = [sys.executable, str(Path(__file__).parent / script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n  ▶ {desc}")
    print(f"    Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ❌ {script} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  ✅ {desc} — done.")


def confirm(message: str, auto_yes: bool) -> bool:
    if auto_yes:
        print(f"  [auto-yes] {message}")
        return True
    resp = input(f"\n  {message} [y/N] ").strip().lower()
    return resp in ("y", "yes")


def show_menu():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           ADDITIVE PEFT — Bottleneck Adapters & (IA)³            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Check VRAM requirements                                      ║
║     Estimates GPU memory for your chosen method and config       ║
║                                                                  ║
║  2. Prepare data                                                 ║
║     Load, format, tokenize dataset with response masking         ║
║                                                                  ║
║  3. Train                                                        ║
║     Fine-tune with Bottleneck Adapters or (IA)³                  ║
║                                                                  ║
║  4. Run inference                                                ║
║     Test the trained adapter on a prompt                         ║
║                                                                  ║
║  5. Compare / analyze                                            ║
║     Inspect what the adapter learned (weight drift, l vectors)   ║
║                                                                  ║
║  6. Full pipeline (1→5)                                          ║
║                                                                  ║
║  7. Switch method (bottleneck ↔ ia3)                             ║
║                                                                  ║
║  Q. Quit                                                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def show_method_explainer(method: str):
    """Print a reminder of what the chosen method does."""
    print("\n" + "─" * 60)
    if method == "bottleneck":
        print("""
  BOTTLENECK ADAPTERS — what is happening:

  Architecture change:
    New modules inserted IN SERIES after each FFN block:
    h → [frozen FFN] → [LN] → [W_down → GELU → W_up] + h → ...
                               ↑ new trainable module, always present

  Key properties:
    • Non-linear (GELU inside) → most expressive of all PEFT methods
    • CANNOT merge into base model (non-linearity blocks it)
    • Permanent inference overhead: ~5–15% slower per token
    • Best for: large domain shifts (e.g., English → medical jargon)

  Initialization (identity trick):
    W_up = zeros  →  output = h + 0 = h  at step 0 (transparent)
    W_down = random  →  learns to compress useful signal as B trains
""")
    else:
        print("""
  (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations:

  Architecture change:
    Tiny learned vectors (l_k, l_v, l_ff) multiplied into existing activations:
    K = (l_k ⊙ W_k) · h       ← amplify/suppress key features
    V = (l_v ⊙ W_v) · h       ← amplify/suppress value features
    FFN = W_down·(l_ff ⊙ GELU(gate)) × up  ← gate FFN channel importance

  Key properties:
    • Fewest parameters of ANY PEFT method (~0.01% of model)
    • Linear (no non-linearity) → CAN merge into base weights (optional)
    • Near-zero inference overhead (element-wise multiply)
    • Best for: lightweight steering, many-task serving, tiny storage budget
    • Less expressive than Bottleneck or LoRA for large domain shifts

  Initialization (identity trick):
    l vectors = ones  →  l ⊙ activations = activations  at step 0 (transparent)
""")
    print("─" * 60)


def get_current_method() -> str:
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("adapter_method", "bottleneck")
    except Exception:
        return "bottleneck"


def set_method(method: str):
    """Update adapter_method in config file."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["adapter_method"] = method
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Config updated: adapter_method = {method}")


def run_pipeline(steps: list[str], prompt: str = None, auto_yes: bool = False):
    method = get_current_method()
    show_method_explainer(method)

    output_dir = f"./outputs/llama-additive-{method}"

    # ── Model discovery: runs once, only when a train step is requested ───────
    #
    # We search for previously saved adapters across three directory scopes
    # (cwd, parent, grandparent) so the user is never silently overwriting work.
    #
    # Decision tree:
    #   found AND user picks "use"     → remove "train" from steps, redirect
    #                                    inference/compare to the chosen path
    #   found AND user picks "retrain" → keep "train", overwrite confirmed
    #   found AND user cancels         → abort pipeline
    #   not found                      → simple y/N confirmation (first run)
    # ─────────────────────────────────────────────────────────────────────────
    effective_adapter_path = output_dir   # may be overridden below

    if "train" in steps:
        found = find_trained_models(output_dir)
        if found:
            decision, chosen_path = prompt_use_or_retrain(
                found, output_dir, auto_yes=auto_yes
            )
            if decision == "use_existing":
                steps = [s for s in steps if s != "train"]
                effective_adapter_path = str(chosen_path)
                print(f"\n  ℹ️  Training skipped.  All subsequent steps will use:")
                print(f"     {effective_adapter_path}")
            elif decision == "cancelled":
                print("  Pipeline cancelled.")
                return
            # "retrain" → fall through; Trainer will overwrite output_dir as normal
        else:
            # No existing model detected — standard first-run gate
            print(f"\n  ℹ️  No existing trained model found in nearby directories.")
            if not auto_yes:
                if not confirm(f"Start training with {method}?", auto_yes):
                    print("  Skipping training.")
                    steps = [s for s in steps if s != "train"]

    step_map = {
        "vram":      ("additive_check_vram.py",    [],                                "VRAM check"),
        "prepare":   ("additive_prepare_data.py",  [],                                "Data preparation"),
        "train":     ("additive_train.py",         [],                                f"Training ({method})"),
        "inference": ("additive_inference.py",
                      ["--adapter_path", effective_adapter_path]
                      + (["--prompt", prompt] if prompt else []),
                      "Inference"),
        "compare":   ("additive_compare.py",
                      ["--adapter_path", effective_adapter_path],
                      "Adapter analysis"),
    }

    for step in steps:
        if step not in step_map:
            print(f"  Unknown step: {step}")
            continue
        script, args, desc = step_map[step]
        run_step(script, args, desc)


def interactive_menu():
    while True:
        show_menu()
        method = get_current_method()
        print(f"  Current method: {method.upper()}")
        choice = input("\n  Enter choice: ").strip().lower()

        if choice == "1":
            run_pipeline(["vram"])
        elif choice == "2":
            run_pipeline(["prepare"])
        elif choice == "3":
            run_pipeline(["train"])
        elif choice == "4":
            prompt = input("  Enter prompt (or press Enter for default): ").strip() or None
            run_pipeline(["inference"], prompt=prompt)
        elif choice == "5":
            run_pipeline(["compare"])
        elif choice == "6":
            run_pipeline(["vram", "prepare", "train", "inference", "compare"])
        elif choice == "7":
            new_method = "ia3" if method == "bottleneck" else "bottleneck"
            set_method(new_method)
            print(f"  Switched to: {new_method.upper()}")
        elif choice in ("q", "quit", "exit"):
            print("  Goodbye.")
            break
        else:
            print("  Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Additive PEFT master controller")
    parser.add_argument("--run",
                        choices=["all", "vram", "prepare", "train", "inference", "compare"],
                        help="Step(s) to run non-interactively")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Inference prompt (used when --run includes inference)")
    parser.add_argument("--method", choices=["bottleneck", "ia3"], default=None,
                        help="Override adapter_method in config")
    parser.add_argument("--yes", action="store_true",
                        help="Auto-confirm all prompts (non-interactive)")
    args = parser.parse_args()

    if args.method:
        set_method(args.method)

    print("\n" + "=" * 60)
    print("  ADDITIVE PEFT — Bottleneck Adapters & (IA)³")
    print("=" * 60)
    print("""
  Additive PEFT inserts new trainable components INTO the model:

  Bottleneck Adapters:
    New sequential module after each FFN:   h → [W_down→GELU→W_up] + h
    Non-linear → most expressive, cannot merge, permanent overhead

  (IA)³:
    Learned scaling vectors on K, V, FFN gates:   K = (l_k ⊙ W_k) · h
    Linear → can merge, near-zero overhead, fewest params of any PEFT
""")

    if args.run is None:
        interactive_menu()
    elif args.run == "all":
        run_pipeline(["vram", "prepare", "train", "inference", "compare"],
                     prompt=args.prompt, auto_yes=args.yes)
    else:
        run_pipeline([args.run], prompt=args.prompt, auto_yes=args.yes)


if __name__ == "__main__":
    main()



# """
# additive_main.py — Master controller for Additive PEFT (Bottleneck & (IA)³).
#
# Mirrors the structure of peft_main.py exactly, with additive-specific steps.
#
# Usage:
#     python additive_main.py                     # Interactive menu
#     python additive_main.py --run all           # Full pipeline
#     python additive_main.py --run train         # Just train
#     python additive_main.py --run vram          # Just check VRAM
#     python additive_main.py --run compare       # Analyze what adapter learned
#     python additive_main.py --prompt "..."      # Custom inference prompt
#     python additive_main.py --method ia3        # Switch to (IA)³ (overrides config)
#     python additive_main.py --run all --yes     # Auto-confirm all prompts
# """
#
# import argparse
# import subprocess
# import sys
# import yaml
# from pathlib import Path
#
# CONFIG_PATH = Path(__file__).parent / "additive_training_config.yaml"
#
#
# def run_step(script: str, extra_args: list[str] = None, desc: str = ""):
#     """Run a pipeline step as a subprocess."""
#     cmd = [sys.executable, str(Path(__file__).parent / script)]
#     if extra_args:
#         cmd.extend(extra_args)
#     print(f"\n  ▶ {desc}")
#     print(f"    Running: {' '.join(cmd)}\n")
#     result = subprocess.run(cmd)
#     if result.returncode != 0:
#         print(f"\n  ❌ {script} failed (exit code {result.returncode})")
#         sys.exit(result.returncode)
#     print(f"\n  ✅ {desc} — done.")
#
#
# def confirm(message: str, auto_yes: bool) -> bool:
#     if auto_yes:
#         print(f"  [auto-yes] {message}")
#         return True
#     resp = input(f"\n  {message} [y/N] ").strip().lower()
#     return resp in ("y", "yes")
#
#
# def show_menu():
#     print("""
# ╔══════════════════════════════════════════════════════════════════╗
# ║           ADDITIVE PEFT — Bottleneck Adapters & (IA)³            ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║                                                                  ║
# ║  1. Check VRAM requirements                                      ║
# ║     Estimates GPU memory for your chosen method and config       ║
# ║                                                                  ║
# ║  2. Prepare data                                                 ║
# ║     Load, format, tokenize dataset with response masking         ║
# ║                                                                  ║
# ║  3. Train                                                        ║
# ║     Fine-tune with Bottleneck Adapters or (IA)³                  ║
# ║                                                                  ║
# ║  4. Run inference                                                ║
# ║     Test the trained adapter on a prompt                         ║
# ║                                                                  ║
# ║  5. Compare / analyze                                            ║
# ║     Inspect what the adapter learned (weight drift, l vectors)   ║
# ║                                                                  ║
# ║  6. Full pipeline (1→5)                                          ║
# ║                                                                  ║
# ║  7. Switch method (bottleneck ↔ ia3)                             ║
# ║                                                                  ║
# ║  Q. Quit                                                         ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
# """)
#
#
# def show_method_explainer(method: str):
#     """Print a reminder of what the chosen method does."""
#     print("\n" + "─" * 60)
#     if method == "bottleneck":
#         print("""
#   BOTTLENECK ADAPTERS — what is happening:
#
#   Architecture change:
#     New modules inserted IN SERIES after each FFN block:
#     h → [frozen FFN] → [LN] → [W_down → GELU → W_up] + h → ...
#                                ↑ new trainable module, always present
#
#   Key properties:
#     • Non-linear (GELU inside) → most expressive of all PEFT methods
#     • CANNOT merge into base model (non-linearity blocks it)
#     • Permanent inference overhead: ~5–15% slower per token
#     • Best for: large domain shifts (e.g., English → medical jargon)
#
#   Initialization (identity trick):
#     W_up = zeros  →  output = h + 0 = h  at step 0 (transparent)
#     W_down = random  →  learns to compress useful signal as B trains
# """)
#     else:
#         print("""
#   (IA)³ — Infused Adapter by Inhibiting and Amplifying Inner Activations:
#
#   Architecture change:
#     Tiny learned vectors (l_k, l_v, l_ff) multiplied into existing activations:
#     K = (l_k ⊙ W_k) · h       ← amplify/suppress key features
#     V = (l_v ⊙ W_v) · h       ← amplify/suppress value features
#     FFN = W_down·(l_ff ⊙ GELU(gate)) × up  ← gate FFN channel importance
#
#   Key properties:
#     • Fewest parameters of ANY PEFT method (~0.01% of model)
#     • Linear (no non-linearity) → CAN merge into base weights (optional)
#     • Near-zero inference overhead (element-wise multiply)
#     • Best for: lightweight steering, many-task serving, tiny storage budget
#     • Less expressive than Bottleneck or LoRA for large domain shifts
#
#   Initialization (identity trick):
#     l vectors = ones  →  l ⊙ activations = activations  at step 0 (transparent)
# """)
#     print("─" * 60)
#
#
# def get_current_method() -> str:
#     try:
#         with open(CONFIG_PATH) as f:
#             cfg = yaml.safe_load(f)
#         return cfg.get("adapter_method", "bottleneck")
#     except Exception:
#         return "bottleneck"
#
#
# def set_method(method: str):
#     """Update adapter_method in config file."""
#     with open(CONFIG_PATH) as f:
#         cfg = yaml.safe_load(f)
#     cfg["adapter_method"] = method
#     with open(CONFIG_PATH, "w") as f:
#         yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
#     print(f"  Config updated: adapter_method = {method}")
#
#
# def run_pipeline(steps: list[str], prompt: str = None, auto_yes: bool = False):
#     method = get_current_method()
#     show_method_explainer(method)
#
#     output_dir = f"./outputs/llama-additive-{method}"
#
#     step_map = {
#         "vram":      ("additive_check_vram.py",    [],                       "VRAM check"),
#         "prepare":   ("additive_prepare_data.py",  [],                       "Data preparation"),
#         "train":     ("additive_train.py",         [],                       f"Training ({method})"),
#         "inference": ("additive_inference.py",
#                       ["--adapter_path", output_dir]
#                       + (["--prompt", prompt] if prompt else []),
#                       "Inference"),
#         "compare":   ("additive_compare.py",
#                       ["--adapter_path", output_dir],
#                       "Adapter analysis"),
#     }
#
#     for step in steps:
#         if step not in step_map:
#             print(f"  Unknown step: {step}")
#             continue
#         script, args, desc = step_map[step]
#         if step == "train" and not auto_yes:
#             if not confirm(f"Start training with {method}?", auto_yes):
#                 print("  Skipping training.")
#                 continue
#         run_step(script, args, desc)
#
#
# def interactive_menu():
#     while True:
#         show_menu()
#         method = get_current_method()
#         print(f"  Current method: {method.upper()}")
#         choice = input("\n  Enter choice: ").strip().lower()
#
#         if choice == "1":
#             run_pipeline(["vram"])
#         elif choice == "2":
#             run_pipeline(["prepare"])
#         elif choice == "3":
#             run_pipeline(["train"])
#         elif choice == "4":
#             prompt = input("  Enter prompt (or press Enter for default): ").strip() or None
#             run_pipeline(["inference"], prompt=prompt)
#         elif choice == "5":
#             run_pipeline(["compare"])
#         elif choice == "6":
#             run_pipeline(["vram", "prepare", "train", "inference", "compare"])
#         elif choice == "7":
#             new_method = "ia3" if method == "bottleneck" else "bottleneck"
#             set_method(new_method)
#             print(f"  Switched to: {new_method.upper()}")
#         elif choice in ("q", "quit", "exit"):
#             print("  Goodbye.")
#             break
#         else:
#             print("  Invalid choice.")
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Additive PEFT master controller")
#     parser.add_argument("--run",
#                         choices=["all", "vram", "prepare", "train", "inference", "compare"],
#                         help="Step(s) to run non-interactively")
#     parser.add_argument("--prompt", type=str, default=None,
#                         help="Inference prompt (used when --run includes inference)")
#     parser.add_argument("--method", choices=["bottleneck", "ia3"], default=None,
#                         help="Override adapter_method in config")
#     parser.add_argument("--yes", action="store_true",
#                         help="Auto-confirm all prompts (non-interactive)")
#     args = parser.parse_args()
#
#     if args.method:
#         set_method(args.method)
#
#     print("\n" + "=" * 60)
#     print("  ADDITIVE PEFT — Bottleneck Adapters & (IA)³")
#     print("=" * 60)
#     print("""
#   Additive PEFT inserts new trainable components INTO the model:
#
#   Bottleneck Adapters:
#     New sequential module after each FFN:   h → [W_down→GELU→W_up] + h
#     Non-linear → most expressive, cannot merge, permanent overhead
#
#   (IA)³:
#     Learned scaling vectors on K, V, FFN gates:   K = (l_k ⊙ W_k) · h
#     Linear → can merge, near-zero overhead, fewest params of any PEFT
# """)
#
#     if args.run is None:
#         interactive_menu()
#     elif args.run == "all":
#         run_pipeline(["vram", "prepare", "train", "inference", "compare"],
#                      prompt=args.prompt, auto_yes=args.yes)
#     else:
#         run_pipeline([args.run], prompt=args.prompt, auto_yes=args.yes)
#
#
# if __name__ == "__main__":
#     main()
