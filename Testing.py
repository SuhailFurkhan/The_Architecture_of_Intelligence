"""
Fine Tuning - Detailed Breakdown
============================================

[Optional: longer overview paragraph you can fill in later]
"""

import os
import sys
import subprocess
from pathlib import Path

TOPIC_NAME = "Fine Tuning_Detailed Breakdown"

# ─────────────────────────────────────────────────────────────────────────────
# PATH TO THE PIPELINE SCRIPT
# Adjust this to match your actual project layout
# ─────────────────────────────────────────────────────────────────────────────

# This resolves relative to this file's location:
#   topics/08_a_FineTuning_FullFineTuning.py
#   Implementation/Full_Fine_Tuning_Implementation/scripts/Full_fine_tuning_main.py
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "Implementation" / "Full_Fine_Tuning_Implementation" / "scripts"
_MAIN_SCRIPT = _SCRIPTS_DIR / "Full_fine_tuning_main.py"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY  (unchanged — keeping your existing content)
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Fine Tuning Detailed Breakdown

                                                                            FINE-TUNING METHODS HIERARCHY — LANDSCAPE VIEW

                                                ══════════════════════════════════════════════════════════════════════════════════════════════════════

                                                                                            ┌──────────────────────┐
                                                                                            │     FINE-TUNING      │
                                                                                            │  (Adapting a model   │
                                                                                            │   to a specific task)│
                                                                                            └──────────┬───────────┘
                                                                                                       │
                                  ┌────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┐
                                  │                                                                    │                                                                    │
                                  ▼                                                                    ▼                                                                    ▼
                  ┌───────────────────────────────┐                              ┌────────────────────────────────────────┐                             ┌─────────────────────────────────────┐
                  │       FULL FINE-TUNING        │                              │    PEFT (Parameter-Efficient           │                             │       ALIGNMENT TUNING              │
                  │                               │                              │    Fine-Tuning)                        │                             │    (Human Preference-Based)         │
                  │  • ALL params updated         │                              │                                        │                             │                                     │
                  │  • Best quality potential     │                              │  • Only a SUBSET of params updated     │                             │  • Aligns model behavior            │
                  │  • Highest cost (GPU/memory)  │                              │  • Lower cost (memory & compute)       │                             │    with human values                │
                  │  • Risk of catastrophic       │                              │  • Preserves pre-trained knowledge     │                             │  • Uses ranked preferences          │
                  │    forgetting                 │                              │  • Modular (swap adapters per task)    │                             │    or reward signals                │
                  └───────────┬───────────────────┘                              └──────────────────────┬─────────────────┘                             └─────────────────────┬───────────────┘
                              │                                                                         │                                                                     │
              ┌───────────────┼───────────────┐                                                         │                                                                     │
              ▼               ▼               ▼                                                         │                                                                     │
      ┌──────────────┐┌──────────────┐┌──────────────┐                                                  │                                                                     │
      │  Standard    ││  Feature     ││  Gradual     │                                                  │                                                                     │
      │  Full FT     ││  Extraction  ││  Unfreezing  │                                                  │                                                                     │
      │              ││              ││              │                                                  │                                                                     │
      │ All layers   ││ Freeze base, ││ Unfreeze     │                                                  │                                                                     │
      │ unlocked     ││ train new    ││ layers one   │                                                  │                                                                     │
      │ from start   ││ head only    ││ by one       │                                                  │                                                                     │
      └──────────────┘└──────────────┘└──────────────┘                                                  │                                                                     │
                                                                                                        │                                                                     │
                                                                                                        │                                                                     │
                    ┌──────────────────────────────┬──────────────────────────────┬─────────────────────┼──────────────────┬──────────────────────────────┐                   │
                    │                              │                              │                     │                  │                              │                   │
                    ▼                              ▼                              ▼                     │                  ▼                              ▼                   │
        ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐       │      ┌───────────────────────────┐ ┌───────────────────────────┐    │
        │    ADDITIVE METHODS       │ │   REPARAMETERIZATION      │ │    SELECTIVE METHODS      │       │      │     HYBRID METHODS        │ │     PROMPT METHODS        │    │
        │                           │ │                           │ │                           │       │      │                           │ │                           │    │
        │  Add NEW parameters       │ │  Transform existing       │ │  Select WHICH existing    │       │      │  Combine multiple PEFT    │ │  Learn soft prompts,      │    │
        │  to the model while       │ │  params via low-rank      │ │  params to train and      │       │      │  strategies (e.g.         │ │  NOT weights. Trainable   │    │
        │  freezing originals       │ │  decomposition            │ │  freeze the rest          │       │      │  quantization + adapters) │ │  tokens prepended to input│    │
        └─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘       │      └─────────────┬─────────────┘ └──────────────┬────────────┘    │
                      │                             │                             │                     │                    │                              │                 │
                      │                             │                             │                     │                    │                              │                 │
                      ▼                             ▼                             ▼                     │                    ▼                              ▼                 │
                                                                                                       │                                                                    │
                 (See individual topic modules for each method)                                         │                                                                    │
                                                                                                       │                                                                    │
                                                                                                       │                                                                    │
                                                                                                       │                                                                    │
                                                                                                       │                                                                    │
                                                                                                       │                                                                    │
                                                                                                       │                                                                    │
                                                                                                       │                                                                    │
                                                                                                       │                                                                    │
            (Refer to the full version of this chart in the theory for complete details)                │                                                                    │

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
| Aspect          | Detail          |
|-----------------|-----------------|
| Parameters      |                 |
| Training Time   |                 |
| Inference Time  |                 |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Code snippets (these still appear in the standard Operations tab)
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Full Pipeline Overview": {
        "description": "The Full Fine-Tuning pipeline: Token Check → VRAM → Data Prep → Training → Inference → Compare",
        "runnable": False,
        "code": '''# Full Fine-Tuning Pipeline Steps
# ================================
# 1. Token Verification  — Validate HuggingFace credentials
# 2. VRAM Check          — Estimate GPU memory requirements
# 3. Data Preparation    — Download, format & tokenize dataset
# 4. Training            — Full fine-tuning (ALL parameters)
# 5. Inference           — Test your fine-tuned model
# 6. Compare             — Side-by-side: original vs fine-tuned
#
# Run from CLI:
#   python Full_fine_tuning_main.py                    # Interactive menu
#   python Full_fine_tuning_main.py --run all          # Full pipeline
#   python Full_fine_tuning_main.py --run train --yes  # Train, auto-confirm
#
# Or use the 🚀 Pipeline Runner tab to run from within Streamlit!
'''
    },

    "Training Configuration": {
        "description": "Key training hyperparameters for full fine-tuning (from training_config.yaml)",
        "runnable": False,
        "code": '''# training_config.yaml — Key Parameters
# ======================================
model_name: "unsloth/Llama-3.2-1B-Instruct"
dataset_name: "yahma/alpaca-cleaned"
max_seq_length: 512

# Batch & Accumulation
per_device_train_batch_size: 1       # Fits in VRAM
gradient_accumulation_steps: 8       # Effective batch = 1 × 8 = 8

# Optimizer & Schedule
learning_rate: 2e-5
weight_decay: 0.01
warmup_ratio: 0.03
lr_scheduler_type: "cosine"

# Training Duration
num_train_epochs: 3                  # ~17,000 steps on 52K examples

# Precision & Memory
bf16: true
gradient_checkpointing: true         # Trades compute for VRAM savings

# Checkpointing
save_strategy: "steps"
save_steps: 500
save_total_limit: 2
'''
    },

    "VRAM Estimation Formula": {
        "description": "How GPU VRAM requirements are estimated for full fine-tuning",
        "runnable": False,
        "code": '''# VRAM Estimation for Full Fine-Tuning
# =====================================
# For a model with P parameters in bf16:
#
# Model Weights:     P × 2 bytes  (bf16 = 2 bytes per param)
# Gradients:         P × 2 bytes  (same dtype as weights)
# Optimizer (AdamW): P × 8 bytes  (2 states × 4 bytes each, fp32)
# Activations:       ~1-4 GB      (depends on seq_len, batch_size)
#
# Example: Llama-3.2-1B (1.24B params)
#   Weights:    1.24B × 2 = 2.48 GB
#   Gradients:  1.24B × 2 = 2.48 GB
#   Optimizer:  1.24B × 8 = 9.92 GB
#   Activations: ~1.5 GB (with gradient checkpointing)
#   ─────────────────────────────
#   TOTAL:      ~16.4 GB
#
# gradient_checkpointing=True reduces activation memory by ~60-70%
# at the cost of ~20-30% slower training (recomputes activations)
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PIPELINE RUNNER
# This function renders an interactive UI for running the fine-tuning pipeline
# directly from within the Streamlit app.
# ─────────────────────────────────────────────────────────────────────────────

def render_operations():
    """
    Custom Streamlit UI for running the Full Fine-Tuning pipeline.

    Called by app_Testing.py when this topic is selected, instead of
    the default code-snippet rendering.

    Features:
    - Editable config panel (model, batch size, learning rate, etc.)
    - Run individual pipeline steps via buttons
    - Real-time streaming output display
    - Step status tracking
    """
    import streamlit as st

    # ─── Session State Initialization ───
    if "fft_step_outputs" not in st.session_state:
        st.session_state.fft_step_outputs = {}
    if "fft_step_status" not in st.session_state:
        st.session_state.fft_step_status = {}
    if "fft_running" not in st.session_state:
        st.session_state.fft_running = False

    # ─── Resolve Script Path ───
    script_path = _MAIN_SCRIPT
    scripts_dir = _SCRIPTS_DIR

    if not script_path.exists():
        st.error(
            f"Pipeline script not found at:\n`{script_path}`\n\n"
            f"Please verify the path in `08_a_FineTuning_FullFineTuning.py` "
            f"(variables `_SCRIPTS_DIR` and `_MAIN_SCRIPT`)."
        )
        # Still show the standard operations as fallback
        _render_standard_operations(st)
        return

    # ─── Layout ───
    config_tab, runner_tab, code_tab = st.tabs([
        "⚙️ Configuration",
        "🚀 Pipeline Runner",
        "📝 Code Reference"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    with config_tab:
        st.markdown("### Training Configuration")
        st.caption("These values are sent to the pipeline. Edit before running.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model & Data**")
            model_name = st.selectbox(
                "Model",
                options=[
                    "unsloth/Llama-3.2-1B-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    "openai-community/gpt2",
                ],
                index=0,
                key="fft_model_name",
            )
            dataset_name = st.text_input(
                "Dataset", value="yahma/alpaca-cleaned", key="fft_dataset"
            )
            max_seq_length = st.select_slider(
                "Max Sequence Length",
                options=[128, 256, 512, 1024, 2048],
                value=512,
                key="fft_seq_len",
            )

        with col2:
            st.markdown("**Training Hyperparameters**")
            batch_size = st.number_input(
                "Per-Device Batch Size", min_value=1, max_value=16, value=1,
                key="fft_batch_size",
            )
            grad_accum = st.number_input(
                "Gradient Accumulation Steps", min_value=1, max_value=64, value=8,
                key="fft_grad_accum",
            )
            num_epochs = st.number_input(
                "Epochs", min_value=1, max_value=10, value=3,
                key="fft_epochs",
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                value=2e-5,
                format_func=lambda x: f"{x:.0e}",
                key="fft_lr",
            )

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Precision & Memory**")
            use_bf16 = st.checkbox("Use bf16 (Brain Float 16)", value=True, key="fft_bf16")
            use_grad_ckpt = st.checkbox(
                "Gradient Checkpointing (saves VRAM)", value=True, key="fft_grad_ckpt"
            )
            lr_scheduler = st.selectbox(
                "LR Scheduler",
                options=["cosine", "linear", "constant", "constant_with_warmup"],
                index=0,
                key="fft_scheduler",
            )

        with col4:
            st.markdown("**Checkpointing & Logging**")
            logging_steps = st.number_input(
                "Logging Steps", min_value=1, max_value=100, value=10,
                key="fft_log_steps",
            )
            save_steps = st.number_input(
                "Save Checkpoint Every N Steps", min_value=50, max_value=2000, value=500,
                key="fft_save_steps",
            )
            eval_steps = st.number_input(
                "Eval Every N Steps", min_value=50, max_value=2000, value=200,
                key="fft_eval_steps",
            )

        # Show effective batch size
        effective_bs = batch_size * grad_accum
        st.info(f"**Effective batch size:** {batch_size} × {grad_accum} = **{effective_bs}**")

        # Build config dict (used by the pipeline)
        config = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "max_seq_length": max_seq_length,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": grad_accum,
            "num_train_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lr_scheduler_type": lr_scheduler,
            "bf16": use_bf16,
            "gradient_checkpointing": use_grad_ckpt,
            "output_dir": "./outputs/llama-3.2-1B-full-ft",
            "logging_steps": logging_steps,
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            "save_strategy": "steps",
            "save_steps": save_steps,
            "save_total_limit": 2,
            "seed": 42,
        }

        # Store config in session state for the runner tab
        st.session_state.fft_config = config

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: PIPELINE RUNNER
    # ═══════════════════════════════════════════════════════════════════════
    with runner_tab:
        st.markdown("### Full Fine-Tuning Pipeline")
        st.caption(
            "Run each step individually or the full pipeline. "
            "Output streams in real-time below each step."
        )

        # Pipeline steps definition
        steps = [
            ("token", "1. Verify HF Token", "Validates your HuggingFace credentials and model access"),
            ("vram", "2. Check VRAM", "Estimates GPU memory requirements for your config"),
            ("prepare", "3. Prepare Dataset", "Downloads, formats, and tokenizes the training data"),
            ("train", "4. Train Model", "Full fine-tuning — ⚠️ Takes HOURS (see warning below)"),
            ("inference", "5. Test Inference", "Generate text with the fine-tuned model"),
            ("compare", "6. Compare Models", "Side-by-side comparison: original vs fine-tuned"),
        ]

        # ── Training Warning ──
        with st.expander("⚠️ Training Time Warning", expanded=False):
            st.warning(
                "**Full fine-tuning is EXTREMELY time-consuming!**\n\n"
                "Estimated time:\n"
                "- RTX 3090 (24 GB): ~3-6 hours\n"
                "- RTX 4090 (24 GB): ~2-4 hours\n"
                "- A100 (40/80 GB): ~1-2 hours\n"
                "- CPU only: Days (not recommended)\n\n"
                "~17,000+ optimizer steps across 3 epochs over 52K examples. "
                "Do NOT close the browser during training."
            )

        st.markdown("---")

        # ── Individual Step Runners ──
        for step_key, step_label, step_desc in steps:
            with st.container(border=True):
                col_info, col_btn = st.columns([3, 1])

                with col_info:
                    # Status indicator
                    status = st.session_state.fft_step_status.get(step_key, "pending")
                    status_icons = {
                        "pending": "⬜",
                        "running": "🔄",
                        "success": "✅",
                        "failed": "❌",
                    }
                    icon = status_icons.get(status, "⬜")
                    st.markdown(f"**{icon} {step_label}**")
                    st.caption(step_desc)

                with col_btn:
                    st.markdown("")  # vertical spacer
                    # Extra confirmation for training step
                    if step_key == "train":
                        confirm_train = st.checkbox(
                            "I understand this takes hours",
                            key="fft_confirm_train",
                        )
                        run_disabled = not confirm_train
                    else:
                        run_disabled = False

                    if st.button(
                            f"Run",
                            key=f"fft_run_{step_key}",
                            use_container_width=True,
                            disabled=run_disabled,
                            type="primary" if step_key == "train" else "secondary",
                    ):
                        _execute_step(st, step_key, step_label, script_path, scripts_dir)

                # Show output if available
                if step_key in st.session_state.fft_step_outputs:
                    output = st.session_state.fft_step_outputs[step_key]
                    with st.expander(f"📋 Output from {step_label}", expanded=True):
                        st.code(output, language="text")

        # ── Full Pipeline Button ──
        st.markdown("---")
        st.markdown("### Run Full Pipeline")
        st.caption("Runs steps 1 → 6 sequentially. Each step must succeed before the next starts.")

        confirm_full = st.checkbox(
            "I understand this will take several hours and I've configured everything above",
            key="fft_confirm_full",
        )

        if st.button(
                "🚀 Run Full Pipeline",
                key="fft_run_all",
                disabled=not confirm_full,
                type="primary",
                use_container_width=True,
        ):
            _execute_full_pipeline(st, steps, script_path, scripts_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: CODE REFERENCE (Standard operations display)
    # ═══════════════════════════════════════════════════════════════════════
    with code_tab:
        _render_standard_operations(st)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Execute a single pipeline step via subprocess
# ─────────────────────────────────────────────────────────────────────────────

def _execute_step(st, step_key, step_label, script_path, scripts_dir):
    """
    Run a single pipeline step as a subprocess and stream output to Streamlit.

    Uses subprocess.Popen to launch:
        python Full_fine_tuning_main.py --run <step_key> --yes

    Output is captured line-by-line and displayed in real-time.
    """
    st.session_state.fft_step_status[step_key] = "running"

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--run", step_key,
        "--yes",  # Auto-confirm prompts (non-interactive mode)
    ]

    # For inference, add a default prompt
    if step_key == "inference":
        cmd.extend(["--prompt", "What is machine learning? Explain in 2 sentences."])

    output_lines = []
    output_placeholder = st.empty()

    try:
        output_placeholder.info(f"🔄 Running {step_label}...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
            cwd=str(scripts_dir),  # So local imports (check_vram, etc.) resolve
            env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Force unbuffered output
        )

        # Stream output line-by-line
        for line in process.stdout:
            # Strip ANSI escape codes for clean display
            clean_line = _strip_ansi(line)
            output_lines.append(clean_line)
            # Update the display with accumulated output
            output_placeholder.code("".join(output_lines), language="text")

        process.wait()

        if process.returncode == 0:
            st.session_state.fft_step_status[step_key] = "success"
            output_lines.append(f"\n{'=' * 50}\n✅ {step_label} completed successfully.\n")
        else:
            st.session_state.fft_step_status[step_key] = "failed"
            output_lines.append(
                f"\n{'=' * 50}\n❌ {step_label} failed (exit code {process.returncode}).\n"
            )

    except FileNotFoundError:
        st.session_state.fft_step_status[step_key] = "failed"
        output_lines.append(f"❌ Could not find Python or script at:\n  {script_path}\n")
    except Exception as e:
        st.session_state.fft_step_status[step_key] = "failed"
        output_lines.append(f"❌ Error: {e}\n")

    # Store final output
    final_output = "".join(output_lines)
    st.session_state.fft_step_outputs[step_key] = final_output
    output_placeholder.code(final_output, language="text")


def _execute_full_pipeline(st, steps, script_path, scripts_dir):
    """Run all pipeline steps sequentially, stopping on failure."""
    progress_bar = st.progress(0, text="Starting pipeline...")
    total_steps = len(steps)

    for i, (step_key, step_label, _) in enumerate(steps):
        progress_bar.progress(
            (i) / total_steps,
            text=f"Running {step_label} ({i + 1}/{total_steps})..."
        )

        _execute_step(st, step_key, step_label, script_path, scripts_dir)

        if st.session_state.fft_step_status.get(step_key) == "failed":
            progress_bar.progress(
                (i + 1) / total_steps,
                text=f"❌ Pipeline stopped at {step_label}"
            )
            st.error(f"Pipeline stopped: {step_label} failed. Fix the issue and retry.")
            return

    progress_bar.progress(1.0, text="✅ Full pipeline completed!")
    st.success("All pipeline steps completed successfully!")
    st.balloons()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Render standard OPERATIONS dict (fallback / code reference tab)
# ─────────────────────────────────────────────────────────────────────────────

def _render_standard_operations(st):
    """Render the OPERATIONS dict in standard expander format."""
    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**Description:** {op_data['description']}")
            st.markdown("---")
            st.code(op_data["code"], language="python")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Strip ANSI escape codes from terminal output
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    """Remove ANSI color/formatting codes from text."""
    import re
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    return {
        "theory": THEORY,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
        "render_operations": render_operations,  # Custom Streamlit UI for pipeline
    }
