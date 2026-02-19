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

    step_map = {
        "vram":      ("additive_check_vram.py",    [],                       "VRAM check"),
        "prepare":   ("additive_prepare_data.py",  [],                       "Data preparation"),
        "train":     ("additive_train.py",         [],                       f"Training ({method})"),
        "inference": ("additive_inference.py",
                      ["--adapter_path", output_dir]
                      + (["--prompt", prompt] if prompt else []),
                      "Inference"),
        "compare":   ("additive_compare.py",
                      ["--adapter_path", output_dir],
                      "Adapter analysis"),
    }

    for step in steps:
        if step not in step_map:
            print(f"  Unknown step: {step}")
            continue
        script, args, desc = step_map[step]
        if step == "train" and not auto_yes:
            if not confirm(f"Start training with {method}?", auto_yes):
                print("  Skipping training.")
                continue
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
