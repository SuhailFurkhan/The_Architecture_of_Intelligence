"""
AI Concepts Reference Hub
==========================
A Streamlit application to explain AI/ML concepts, theory, and implementations
from Perceptron to Large Language Models.

Structure:
- app.py (this file) - Main Streamlit application
- topics/ - Package containing topic modules (auto-discovered)
- Implementation/ - Folder containing concept implementation .py files
- LLM_module.py - AI assistant backend
- SolutionGeneration.py - Vision-based analysis
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as st_components
from topics import get_all_topics
from Automation_Infrastructure import get_all_tutorials

# LLM Module for AI Assistant
from LLM_module import (
    LLMAssistant,
    LLMResponse,
    get_available_providers,
    format_code_context,
    EXAMPLE_QUERIES,
    get_api_key_from_env,
    load_env_file
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Concepts Reference Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: "Times New Roman", Times, serif;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD TOPICS AND IMPLEMENTATIONS
# =============================================================================

@st.cache_data
def load_topics():
    """Load all topics from the topics package."""
    return get_all_topics()


@st.cache_data
def load_implementations():
    """
    Load all implementation .py files from the Implementation folder.

    Expected file format with metadata in docstring:
    ```
    \"\"\"
    Perceptron: Binary Classifier from Scratch
    Level: Beginner
    Concepts: Perceptron, Gradient Descent
    ...
    \"\"\"
    ```

    Returns:
        dict: {key: {"display_name": str, "code": str, "path": str,
                      "level": str, "concepts": list}}
    """
    base_dir = Path(__file__).parent
    impl_dir = base_dir / "Implementation"

    if not impl_dir.exists() or not impl_dir.is_dir():
        return {}

    implementations = {}
    for py_file in sorted(impl_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        try:
            code_text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        key = py_file.stem
        display_name = key.replace("_", " ").title()

        # Parse metadata from docstring
        level = "Unknown"
        concepts = []

        for line in code_text.split("\n"):
            line_stripped = line.strip()

            if line_stripped.lower().startswith("level:"):
                level = line_stripped.split(":", 1)[1].strip()
                level_lower = level.lower()
                if "beginner" in level_lower:
                    level = "Beginner"
                elif "intermediate" in level_lower:
                    level = "Intermediate"
                elif "advanced" in level_lower:
                    level = "Advanced"

            elif line_stripped.lower().startswith("concepts:"):
                concepts_str = line_stripped.split(":", 1)[1].strip()
                concepts = [c.strip() for c in concepts_str.split(",") if c.strip()]

        implementations[key] = {
            "display_name": display_name,
            "code": code_text,
            "path": str(py_file),
            "level": level,
            "concepts": concepts,
        }

    return implementations


def get_all_concepts(implementations: dict) -> list:
    """Extract all unique concepts from implementations."""
    all_concepts = set()
    for impl in implementations.values():
        all_concepts.update(impl.get("concepts", []))
    return sorted(all_concepts)


def get_all_levels(implementations: dict) -> list:
    """Extract all unique levels from implementations."""
    levels = set()
    for impl in implementations.values():
        lvl = impl.get("level", "Unknown")
        if lvl:
            levels.add(lvl)
    order = ["Beginner", "Intermediate", "Advanced", "Unknown"]
    return [l for l in order if l in levels]


# Load content
CONTENT = load_topics()
TOPIC_LIST = list(CONTENT.keys()) if CONTENT else []

IMPLEMENTATIONS = load_implementations()
IMPL_KEYS = list(IMPLEMENTATIONS.keys()) if IMPLEMENTATIONS else []


# =============================================================================
# LOAD TUTORIALS (Automation / Infrastructure)
# =============================================================================

@st.cache_data
def load_tutorials():
    """Load all tutorials from the Automation_Infrastructure package."""
    return get_all_tutorials()


TUTORIALS = load_tutorials()
TUTORIAL_LIST = list(TUTORIALS.keys()) if TUTORIALS else []


# =============================================================================
# LOAD API KEY FROM ENV FILE
# =============================================================================

@st.cache_resource
def load_api_keys():
    """Load API keys from Keys.env file."""
    env_vars = load_env_file()
    return env_vars


ENV_KEYS = load_api_keys()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "highlighted_operation" not in st.session_state:
    st.session_state.highlighted_operation = None
if "show_operation_panel" not in st.session_state:
    st.session_state.show_operation_panel = False
if "target_topic" not in st.session_state:
    st.session_state.target_topic = None
if "topic_radio" not in st.session_state and TOPIC_LIST:
    st.session_state.topic_radio = TOPIC_LIST[0]
if "impl_key" not in st.session_state and IMPL_KEYS:
    st.session_state.impl_key = IMPL_KEYS[0]
if "tutorial_radio" not in st.session_state and TUTORIAL_LIST:
    st.session_state.tutorial_radio = TUTORIAL_LIST[0]

# Track which main section to show
if "main_view" not in st.session_state:
    st.session_state.main_view = "topics"  # "topics", "implementation", "tutorials", or "ai_assistant"

# AI Assistant session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm_provider" not in st.session_state:
    if ENV_KEYS.get("ANTHROPIC_API_KEY"):
        st.session_state.llm_provider = "anthropic"
    else:
        st.session_state.llm_provider = "mock"
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = ENV_KEYS.get("ANTHROPIC_API_KEY", "")
if "llm_assistant" not in st.session_state:
    st.session_state.llm_assistant = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "font_size" not in st.session_state:
    st.session_state.font_size = 16  # default in px


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .stExpander {
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .stCodeBlock {
        border-radius: 8px;
    }
    /* Scrollable implementation list */
    .impl-scroll-container {
        max-height: 450px;
        overflow-y: auto;
        padding-right: 5px;
    }
    .impl-scroll-container::-webkit-scrollbar {
        width: 6px;
    }
    .impl-scroll-container::-webkit-scrollbar-track {
        background: #1e1e1e;
        border-radius: 3px;
    }
    .impl-scroll-container::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 3px;
    }
    .impl-scroll-container::-webkit-scrollbar-thumb:hover {
        background: #777;
    }
</style>
""", unsafe_allow_html=True)

# ── Dynamic font-size CSS (driven by slider) ────────────────────────────────
_fs = st.session_state.font_size
st.markdown(f"""
<style>
    /* Scale main content area text */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown td, .stMarkdown th {{
        font-size: {_fs}px !important;
        line-height: 1.6 !important;
    }}
    /* Scale headings proportionally */
    .stMarkdown h1 {{ font-size: {_fs * 2.0:.0f}px !important; }}
    .stMarkdown h2 {{ font-size: {_fs * 1.6:.0f}px !important; }}
    .stMarkdown h3 {{ font-size: {_fs * 1.3:.0f}px !important; }}
    .stMarkdown h4 {{ font-size: {_fs * 1.1:.0f}px !important; }}
    /* Code blocks */
    .stCodeBlock, .stCodeBlock code {{
        font-size: {max(_fs - 2, 12)}px !important;
    }}
    /* Chat messages */
    .stChatMessage p {{
        font-size: {_fs}px !important;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CALLBACK FUNCTIONS
# =============================================================================

def navigate_to_operation(topic_name: str, operation_name: str):
    """Navigate to a specific operation from global search."""
    st.session_state.topic_radio = topic_name
    st.session_state.target_topic = topic_name
    st.session_state.highlighted_operation = operation_name
    st.session_state.show_operation_panel = True
    st.session_state.main_view = "topics"


def clear_highlight():
    """Clear the highlighted operation."""
    st.session_state.highlighted_operation = None
    st.session_state.show_operation_panel = False
    st.session_state.target_topic = None


def switch_to_topics():
    st.session_state.main_view = "topics"


def switch_to_implementation():
    st.session_state.main_view = "implementation"


def switch_to_tutorials():
    st.session_state.main_view = "tutorials"


def switch_to_ai_assistant():
    st.session_state.main_view = "ai_assistant"


def clear_chat_history():
    st.session_state.chat_history = []


def initialize_llm_assistant():
    """Initialize or reinitialize the LLM assistant with current settings."""
    provider = st.session_state.llm_provider
    api_key = st.session_state.llm_api_key if st.session_state.llm_api_key else None
    st.session_state.llm_assistant = LLMAssistant(provider=provider, api_key=api_key)


# =============================================================================
# CODE RUNNER (Subprocess-based)
# =============================================================================

def run_code_subprocess(code_string: str, timeout: int = 30) -> dict:
    """
    Execute a code string in an isolated subprocess and capture output.

    Writes the code to a temp file, runs it with the current Python
    interpreter, and returns stdout/stderr.

    Args:
        code_string: The Python code to execute.
        timeout: Max seconds before the process is killed.

    Returns:
        dict with keys: success (bool), stdout (str), stderr (str)
    """
    tmp_file = None
    try:
        # Write code to a temporary .py file
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        tmp_file.write(code_string)
        tmp_file.flush()
        tmp_file.close()

        # Run in isolated subprocess
        # Force UTF-8 encoding so Unicode chars (✓, ✅, █, etc.) work on Windows
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, tmp_file.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            env=env,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"⏱️ Execution timed out after {timeout} seconds.",
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Runner error: {str(e)}",
        }
    finally:
        # Clean up temp file
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


def render_operation(op_name: str, op_data: dict, key_prefix: str = "op"):
    """
    Render a single operation: description, code block, and a Run button.

    The Run button executes the code in an isolated subprocess and displays
    stdout / stderr directly below the code block inside the same expander.

    Args:
        op_name: Display name of the operation.
        op_data: Dict with 'description', 'code', and optionally 'language'.
        key_prefix: Unique prefix for Streamlit widget keys.
    """
    lang = op_data.get("language", "python")
    st.markdown(f"**Description:** {op_data['description']}")
    st.markdown("---")
    st.code(op_data["code"], language=lang)

    # ── Run Button ────────────────────────────────────────────────────────
    # Create a safe unique key from the operation name
    safe_key = f"{key_prefix}_{op_name}".replace(" ", "_").replace(":", "_")

    # Session-state key that holds this operation's last run result
    result_key = f"run_result_{safe_key}"

    col_run, col_clear, col_spacer = st.columns([1, 1, 3])
    with col_run:
        run_clicked = st.button(
            "▶️ Run",
            key=f"run_{safe_key}",
            type="primary",
            use_container_width=True,
        )
    with col_clear:
        if result_key in st.session_state:
            if st.button("🗑️ Clear", key=f"clear_{safe_key}", use_container_width=True):
                del st.session_state[result_key]
                st.rerun()

    # Execute on click
    if run_clicked:
        with st.spinner("⏳ Running in subprocess..."):
            run_result = run_code_subprocess(op_data["code"])
            st.session_state[result_key] = run_result

    # Display stored result
    if result_key in st.session_state:
        run_result = st.session_state[result_key]
        st.markdown("---")
        st.markdown("#### 📤 Output")

        if run_result["success"]:
            st.success("✅ Execution completed successfully")
        else:
            st.warning("⚠️ Execution finished with errors")

        if run_result["stdout"]:
            st.code(run_result["stdout"], language="text")

        if run_result["stderr"]:
            with st.expander("🔴 Stderr", expanded=not run_result["success"]):
                st.code(run_result["stderr"], language="text")


# =============================================================================
# SIDEBAR
# =============================================================================

# Main Section Selector
st.sidebar.markdown("## 🎯 Main Sections")
col1, col2, col3 = st.sidebar.columns(3) # , col4
with col1:
    st.button("📚 Topics", use_container_width=True,
              type="primary" if st.session_state.main_view == "topics" else "secondary",
              on_click=switch_to_topics)
# with col2:
#     st.button("🔬 Implement", use_container_width=True,
#               type="primary" if st.session_state.main_view == "implementation" else "secondary",
#               on_click=switch_to_implementation)
with col2:
    st.button("🏗️ Infra", use_container_width=True,
              type="primary" if st.session_state.main_view == "tutorials" else "secondary",
              on_click=switch_to_tutorials)
with col3:
    st.button("🤖 AI Help", use_container_width=True,
              type="primary" if st.session_state.main_view == "ai_assistant" else "secondary",
              on_click=switch_to_ai_assistant)

st.sidebar.markdown("---")

# ── Font Size Control ────────────────────────────────────────────────────────
with st.sidebar.expander("🔤 Font Size", expanded=False):
    font_size = st.slider(
        "Adjust text size",
        min_value=12, max_value=28, value=st.session_state.font_size,
        step=1, format="%dpx",
        label_visibility="collapsed"
    )
    st.session_state.font_size = font_size
    _size_label = {12: "XS", 14: "S", 16: "M (default)", 18: "L", 20: "XL", 24: "XXL", 28: "XXXL"}
    st.caption(f"Current: {font_size}px — {_size_label.get(font_size, '')}")

# Show relevant sidebar based on main view
selected_topic = None

if st.session_state.main_view == "topics":
    # ─── Topics Sidebar ─────────────────────────────────────────────
    st.sidebar.markdown("## 📚 AI/ML Topics")

    if CONTENT:
        selected_topic = st.sidebar.radio(
            "Select a topic:",
            TOPIC_LIST,
            label_visibility="collapsed",
            key="topic_radio"
        )

        if st.session_state.target_topic and selected_topic != st.session_state.target_topic:
            clear_highlight()
    else:
        st.sidebar.error("No topics found!")
        selected_topic = None

elif st.session_state.main_view == "implementation":
    # ─── Implementation Sidebar ──────────────────────────────────────
    st.sidebar.markdown("## 🔬 Implementations")

    if IMPLEMENTATIONS:
        all_concepts = get_all_concepts(IMPLEMENTATIONS)
        all_levels = get_all_levels(IMPLEMENTATIONS)

        # Search box
        impl_search = st.sidebar.text_input(
            "🔍 Search implementations",
            placeholder="Search name, concepts...",
            key="impl_search"
        )

        # Level filter
        st.sidebar.markdown("**Level:**")
        level_icons = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴", "Unknown": "⚪"}

        selected_levels = st.sidebar.multiselect(
            "Filter by level",
            options=all_levels,
            default=[],
            format_func=lambda x: f"{level_icons.get(x, '⚪')} {x}",
            key="level_filter",
            label_visibility="collapsed"
        )

        # Concepts filter
        if all_concepts:
            st.sidebar.markdown("**Concepts:**")
            selected_concepts = st.sidebar.multiselect(
                "Filter by concepts",
                options=all_concepts,
                default=[],
                key="concepts_filter",
                label_visibility="collapsed"
            )
        else:
            selected_concepts = []

        st.sidebar.markdown("---")

        # Filter
        filtered_keys = []
        for k in IMPL_KEYS:
            impl = IMPLEMENTATIONS[k]

            if impl_search:
                search_lower = impl_search.lower()
                name_match = search_lower in impl["display_name"].lower()
                concept_match = any(search_lower in c.lower() for c in impl.get("concepts", []))
                if not (name_match or concept_match):
                    continue

            if selected_levels:
                if impl.get("level", "Unknown") not in selected_levels:
                    continue

            if selected_concepts:
                impl_concepts = impl.get("concepts", [])
                if not any(c in impl_concepts for c in selected_concepts):
                    continue

            filtered_keys.append(k)

        # Show filter summary
        total_count = len(IMPL_KEYS)
        filtered_count = len(filtered_keys)
        if impl_search or selected_levels or selected_concepts:
            st.sidebar.caption(f"Showing {filtered_count} of {total_count} implementations")

        if filtered_keys:
            display_items = []
            key_by_display = {}

            for key in filtered_keys:
                impl = IMPLEMENTATIONS[key]
                lvl = impl.get("level", "Unknown")
                lvl_icon = level_icons.get(lvl, "⚪")
                display = f"{lvl_icon} {impl['display_name']}"
                display_items.append(display)
                key_by_display[display] = key

            # Get current selection
            current_key = st.session_state.get("impl_key", filtered_keys[0])
            current_impl = IMPLEMENTATIONS.get(current_key, {})
            current_lvl = current_impl.get("level", "Unknown")
            current_display = f"{level_icons.get(current_lvl, '⚪')} {current_impl.get('display_name', '')}"

            if current_display in display_items:
                default_index = display_items.index(current_display)
            else:
                default_index = 0

            with st.sidebar.container(height=450):
                selected_display = st.radio(
                    "Select implementation:",
                    display_items,
                    index=default_index,
                    label_visibility="collapsed",
                    key="impl_radio"
                )

            st.session_state.impl_key = key_by_display[selected_display]
        else:
            st.sidebar.info("No implementations match the current filters.")
    else:
        st.sidebar.warning(
            "No implementations found.\n\n"
            "Add `.py` files to `Implementation/` folder."
        )

elif st.session_state.main_view == "tutorials":
    # ─── Tutorials Sidebar ───────────────────────────────────────
    st.sidebar.markdown("## 🏗️ Automation & Infra")

    if TUTORIALS:
        selected_tutorial = st.sidebar.radio(
            "Select a tutorial:",
            TUTORIAL_LIST,
            label_visibility="collapsed",
            key="tutorial_radio"
        )
    else:
        st.sidebar.error("No tutorials found!")
        selected_tutorial = None

st.sidebar.markdown("---")

# Quick Links
st.sidebar.markdown("### 🔗 Quick Links")
st.sidebar.markdown("""
- [PyTorch Docs](https://pytorch.org/docs/)
- [Attention Paper](https://arxiv.org/abs/1706.03762)
- [3Blue1Brown Neural Nets](https://www.3blue1brown.com/topics/neural-networks)
- [Andrej Karpathy](https://karpathy.ai/)
- [Hugging Face](https://huggingface.co/)
""")

st.sidebar.markdown("---")

# Help section
with st.sidebar.expander("ℹ️ How to Add Content"):
    st.markdown("""
    **Add Topics:**
    1. Create a new `.py` file in `topics/`
    2. Follow the structure in `topics/learning_path.py`
    3. Include `TOPIC_NAME`, `THEORY`, `COMPLEXITY`, `OPERATIONS`, `get_content()`
    4. Restart the app — auto-discovered!

    **Add Implementations:**
    1. Copy `template.py` into `Implementation/`
    2. Rename (e.g., `perceptron_classifier.py`)
    3. Add `Level:` and `Concepts:` metadata in the docstring
    4. Restart the app

    **Add Infra Tutorials:**
    1. Copy `Automation_Infrastructure/_tutorial_template.py`
    2. Rename (e.g., `docker_compose.py`) — no underscore prefix!
    3. Fill in `TOPIC_NAME`, `CATEGORY`, `THEORY`, `COMMANDS`, `OPERATIONS`
    4. Restart the app — auto-discovered!
    """)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("# 🧠 AI Concepts Reference Hub")

# =========================================================================
# TOPICS VIEW
# =========================================================================
if st.session_state.main_view == "topics":

    if selected_topic and selected_topic in CONTENT:
        topic_data = CONTENT[selected_topic]

        st.markdown(f"## 📌 {selected_topic}")

        # Highlighted Operation Panel (from Global Search)
        highlighted_op = st.session_state.highlighted_operation
        if highlighted_op and st.session_state.show_operation_panel:
            if highlighted_op in topic_data["operations"]:
                op_data = topic_data["operations"][highlighted_op]

                st.markdown("---")
                st.success(f"🎯 **Search Result:** {highlighted_op}")

                with st.container(border=True):
                    st.markdown(f"### {highlighted_op}")
                    render_operation(highlighted_op, op_data, key_prefix="highlight")

                st.button("✕ Close this panel", type="secondary", on_click=clear_highlight)
                st.markdown("---")
                st.caption("👇 Browse all operations in the tabs below")

        # Main Tabs
        tab1, tab2, tab3 = st.tabs(["📖 Theory", "📊 Comparison", "🔧 Code Snippets"])

        with tab1:
            with st.container(border=True):
                # ── Interactive component support ────────────────────────
                # If the topic provides interactive_components, split the
                # raw theory at each placeholder and inject an iframe via
                # st.components.v1.html() between the markdown sections.
                if "interactive_components" in topic_data and topic_data["interactive_components"]:
                    theory_text = topic_data.get("theory_raw", topic_data["theory"])
                    components_map = {
                        ic["placeholder"]: ic
                        for ic in topic_data["interactive_components"]
                    }
                    # Walk through each placeholder and split / render
                    remaining = theory_text
                    for placeholder, comp in components_map.items():
                        if placeholder in remaining:
                            before, after = remaining.split(placeholder, 1)
                            if before.strip():
                                st.markdown(before, unsafe_allow_html=True)
                            st_components.html(
                                comp["html"],
                                height=comp.get("height", 700),
                                scrolling=False,
                            )
                            remaining = after
                    # Render any text after the last placeholder
                    if remaining.strip():
                        st.markdown(remaining, unsafe_allow_html=True)
                else:
                    st.markdown(topic_data["theory"], unsafe_allow_html=True)

        with tab2:
            with st.container(border=True):
                st.markdown(topic_data["complexity"], unsafe_allow_html=True)

        with tab3:
            # Check if this topic provides a custom operations renderer
            custom_renderer = topic_data.get("render_operations")
            if custom_renderer and callable(custom_renderer):
                # Topic has its own Streamlit UI (e.g. Fine-Tuning pipeline runner)
                custom_renderer()
            else:
                # Default: show code snippets in expanders (unchanged)
                search_term = st.text_input(
                    "🔍 Search operations",
                    placeholder="Type to filter operations...",
                    key="operation_search"
                )

                sections = topic_data.get("sections", None)
                all_operations = topic_data["operations"]

                filtered_operations = {}
                for op_name, op_data in all_operations.items():
                    if (search_term.lower() in op_name.lower() or
                            search_term.lower() in op_data["description"].lower()):
                        filtered_operations[op_name] = op_data

                total_ops = len(all_operations)
                filtered_count = len(filtered_operations)

                if search_term:
                    st.caption(f"Showing {filtered_count} of {total_ops} operations")
                else:
                    st.caption(f"{total_ops} operations available • *Click to expand*")

                st.markdown("---")

                auto_expand = bool(search_term) and filtered_count <= 2

                if filtered_operations:
                    if sections and not search_term:
                        for section_name, section_ops in sections.items():
                            section_has_ops = any(op in filtered_operations for op in section_ops)

                            if section_has_ops:
                                st.markdown(f"### {section_name}")

                                for op_name in section_ops:
                                    if op_name in filtered_operations:
                                        op_data = filtered_operations[op_name]
                                        with st.expander(f"▶️ {op_name}", expanded=auto_expand):
                                            st.markdown(f"**Description:** {op_data['description']}")
                                            st.markdown("---")
                                            st.code(op_data["code"], language="python")

                                st.markdown("")

                        # Uncategorized operations
                        ops_in_sections = set()
                        for section_ops in sections.values():
                            ops_in_sections.update(section_ops)

                        uncategorized = {k: v for k, v in filtered_operations.items() if k not in ops_in_sections}
                        if uncategorized:
                            st.markdown("### 📂 Other")
                            for op_name, op_data in uncategorized.items():
                                with st.expander(f"▶️ {op_name}", expanded=auto_expand):
                                    st.markdown(f"**Description:** {op_data['description']}")
                                    st.markdown("---")
                                    st.code(op_data["code"], language="python")
                    else:
                        for operation_name, operation_data in filtered_operations.items():
                            with st.expander(f"▶️ {operation_name}", expanded=auto_expand):
                                st.markdown(f"**Description:** {operation_data['description']}")
                                st.markdown("---")
                                st.code(operation_data["code"], language="python")
                else:
                    st.info(f"No operations found matching '{search_term}'")


    #     with tab3:
    #         search_term = st.text_input(
    #             "🔍 Search code snippets",
    #             placeholder="Type to filter...",
    #             key="operation_search"
    #         )
    #
    #         sections = topic_data.get("sections", None)
    #         all_operations = topic_data["operations"]
    #
    #         filtered_operations = {}
    #         for op_name, op_data in all_operations.items():
    #             if (search_term.lower() in op_name.lower() or
    #                     search_term.lower() in op_data["description"].lower()):
    #                 filtered_operations[op_name] = op_data
    #
    #         total_ops = len(all_operations)
    #         filtered_count = len(filtered_operations)
    #
    #         if search_term:
    #             st.caption(f"Showing {filtered_count} of {total_ops} snippets")
    #         else:
    #             st.caption(f"{total_ops} code snippets available • *Click to expand*")
    #
    #         st.markdown("---")
    #
    #         auto_expand = bool(search_term) and filtered_count <= 2
    #
    #         if filtered_operations:
    #             if sections and not search_term:
    #                 for section_name, section_ops in sections.items():
    #                     section_has_ops = any(op in filtered_operations for op in section_ops)
    #
    #                     if section_has_ops:
    #                         st.markdown(f"### {section_name}")
    #                         for op_name in section_ops:
    #                             if op_name in filtered_operations:
    #                                 op_data = filtered_operations[op_name]
    #                                 with st.expander(f"▶️ {op_name}", expanded=auto_expand):
    #                                     render_operation(op_name, op_data, key_prefix="sec")
    #                         st.markdown("")
    #
    #                 # Uncategorized
    #                 ops_in_sections = set()
    #                 for section_ops in sections.values():
    #                     ops_in_sections.update(section_ops)
    #
    #                 uncategorized = {k: v for k, v in filtered_operations.items() if k not in ops_in_sections}
    #                 if uncategorized:
    #                     st.markdown("### 📂 Other")
    #                     for op_name, op_data in uncategorized.items():
    #                         with st.expander(f"▶️ {op_name}", expanded=auto_expand):
    #                             render_operation(op_name, op_data, key_prefix="uncat")
    #             else:
    #                 for operation_name, operation_data in filtered_operations.items():
    #                     with st.expander(f"▶️ {operation_name}", expanded=auto_expand):
    #                         render_operation(operation_name, operation_data, key_prefix="flat")
    #         else:
    #             st.info(f"No snippets found matching '{search_term}'")
    #
    # elif not CONTENT:
    #     st.error("No topics found! Make sure the `topics/` directory contains valid topic modules.")
    # else:
    #     st.error(f"Topic '{selected_topic}' not found!")


# =========================================================================
# IMPLEMENTATION VIEW
# =========================================================================
elif st.session_state.main_view == "implementation":
    st.markdown("## 🔬 Concept Implementations")

    if IMPLEMENTATIONS and "impl_key" in st.session_state:
        key = st.session_state.impl_key

        if key in IMPLEMENTATIONS:
            impl_entry = IMPLEMENTATIONS[key]

            st.markdown(f"### {impl_entry['display_name']}")

            # Level and Concepts row
            level = impl_entry.get("level", "Unknown")
            concepts = impl_entry.get("concepts", [])

            level_styles = {
                "Beginner": ("🟢", "green"),
                "Intermediate": ("🟡", "orange"),
                "Advanced": ("🔴", "red"),
                "Unknown": ("⚪", "gray")
            }
            lvl_icon, lvl_color = level_styles.get(level, ("⚪", "gray"))

            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                st.markdown(f"**Level:** {lvl_icon} {level}")

            with col2:
                if concepts:
                    concepts_str = " • ".join([f"`{c}`" for c in concepts])
                    st.markdown(f"**Concepts:** {concepts_str}")
                else:
                    st.markdown("**Concepts:** None")

            with col3:
                st.caption(f"📄 `{os.path.basename(impl_entry['path'])}`")

            st.markdown("---")

            # Display code
            with st.container(border=True):
                st.code(impl_entry["code"], language="python")
        else:
            st.info("Select an implementation from the sidebar.")
    else:
        st.info(
            "No implementations loaded.\n\n"
            "**To add implementations:**\n"
            "1. Create an `Implementation/` folder next to `app.py`\n"
            "2. Copy `template.py` and fill in your concept implementation\n"
            "3. Add `Level:` and `Concepts:` in the docstring\n"
            "4. Restart the app"
        )


# =========================================================================
# TUTORIALS VIEW (Automation / Infrastructure)
# =========================================================================
elif st.session_state.main_view == "tutorials":
    st.markdown("## 🏗️ Automation & Infrastructure")

    selected_tutorial = st.session_state.get("tutorial_radio", None)

    if selected_tutorial and selected_tutorial in TUTORIALS:
        tutorial_data = TUTORIALS[selected_tutorial]
        category = tutorial_data.get("category", "General")

        st.markdown(f"## 📌 {selected_tutorial}")
        st.caption(f"Category: `{category}`")

        # Main Tabs
        tab1, tab2, tab3 = st.tabs(["📖 Theory", "📋 Command Reference", "🔧 Step-by-Step"])

        with tab1:
            with st.container(border=True):
                st.markdown(tutorial_data["theory"], unsafe_allow_html=True)

        with tab2:
            with st.container(border=True):
                # st.markdown(tutorial_data.get("commands", "_No command reference yet._"))
                st.markdown(tutorial_data.get("commands", "_No command reference yet._"), unsafe_allow_html=True)

        with tab3:
            search_term = st.text_input(
                "🔍 Search operations",
                placeholder="Type to filter...",
                key="tutorial_operation_search"
            )

            all_operations = tutorial_data.get("operations", {})

            filtered_operations = {}
            for op_name, op_data in all_operations.items():
                if (search_term.lower() in op_name.lower() or
                        search_term.lower() in op_data.get("description", "").lower()):
                    filtered_operations[op_name] = op_data

            total_ops = len(all_operations)
            filtered_count = len(filtered_operations)

            if search_term:
                st.caption(f"Showing {filtered_count} of {total_ops} operations")
            else:
                st.caption(f"{total_ops} operations available • *Click to expand*")

            st.markdown("---")

            auto_expand = bool(search_term) and filtered_count <= 2

            if filtered_operations:
                for op_name, op_data in filtered_operations.items():
                    with st.expander(f"▶️ {op_name}", expanded=auto_expand):
                        st.markdown(f"**Description:** {op_data.get('description', '')}")
                        st.markdown("---")
                        lang = op_data.get("language", "bash")
                        st.code(op_data.get("code", ""), language=lang)
            else:
                st.info(f"No operations found matching '{search_term}'")

    elif not TUTORIALS:
        st.error("No tutorials found! Make sure `Automation_Infrastructure/` has valid modules.")
    else:
        st.info("Select a tutorial from the sidebar.")


# =========================================================================
# AI ASSISTANT VIEW
# =========================================================================
elif st.session_state.main_view == "ai_assistant":
    st.markdown("## 🤖 AI Learning Assistant")
    st.markdown("*Ask questions about neural networks, transformers, training techniques, or any AI/ML concept*")

    # --- Sidebar Configuration for AI ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ AI Configuration")

        has_env_key = bool(ENV_KEYS.get("ANTHROPIC_API_KEY"))
        if has_env_key:
            st.success("✅ API Key loaded from Keys.env")

        available_providers = get_available_providers()
        provider_labels = {
            "mock": "🧪 Mock (Testing)",
            "anthropic": "🔷 Anthropic Claude",
            "openai": "🟢 OpenAI GPT"
        }

        default_provider_idx = 0
        if has_env_key and "anthropic" in available_providers:
            default_provider_idx = available_providers.index("anthropic")

        provider = st.selectbox(
            "Provider",
            options=available_providers,
            index=default_provider_idx,
            format_func=lambda x: provider_labels.get(x, x),
            key="llm_provider_select"
        )

        if provider != st.session_state.llm_provider:
            st.session_state.llm_provider = provider
            st.session_state.llm_assistant = None
            if provider == "anthropic":
                st.session_state.llm_api_key = ENV_KEYS.get("ANTHROPIC_API_KEY", "")
            elif provider == "openai":
                st.session_state.llm_api_key = ENV_KEYS.get("OPENAI_API_KEY", "")

        if provider != "mock":
            current_key = st.session_state.llm_api_key
            if current_key:
                st.caption(f"Using API key: {current_key[:8]}...{current_key[-4:]}")

            with st.expander("🔑 Override API Key"):
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    placeholder=f"Enter your {provider.title()} API key",
                    key="api_key_input"
                )

                if api_key and api_key != st.session_state.llm_api_key:
                    st.session_state.llm_api_key = api_key
                    st.session_state.llm_assistant = None

            if not st.session_state.llm_api_key:
                st.warning("⚠️ No API key found. Add ANTHROPIC_API_KEY to Keys.env")
        else:
            st.info("🧪 Mock mode — no API key needed")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_chat_history()
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Session Info")
        st.caption(f"Messages: {len(st.session_state.chat_history)}")
        st.caption(f"Provider: {provider_labels.get(provider, provider)}")

    # --- Main Chat Interface ---
    st.markdown("---")

    # Example queries
    with st.expander("💡 **Example Questions** — Click to try", expanded=False):
        cols = st.columns(2)
        for i, query in enumerate(EXAMPLE_QUERIES):
            with cols[i % 2]:
                if st.button(f"→ {query[:55]}...", key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_query = query
                    st.rerun()

    # Chat history display
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)

    # Handle pending query from example buttons
    if hasattr(st.session_state, 'pending_query') and st.session_state.pending_query:
        pending = st.session_state.pending_query
        st.session_state.pending_query = None

        st.session_state.chat_history.append({"role": "user", "content": pending})

        if st.session_state.llm_assistant is None:
            initialize_llm_assistant()

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = st.session_state.llm_assistant.query(pending)

                if response.success:
                    st.markdown(response.content)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                else:
                    error_msg = f"❌ Error: {response.error_message}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        st.rerun()

    # Chat input
    user_input = st.chat_input("Ask an AI/ML question...", key="chat_input")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if st.session_state.llm_assistant is None:
            initialize_llm_assistant()

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            try:
                response_placeholder = st.empty()
                full_response = ""

                for chunk in st.session_state.llm_assistant.query_stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        st.rerun()