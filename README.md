# 🧠 AI Concepts Reference Hub

A Streamlit application for learning AI/ML from the ground up — from the Perceptron to Large Language Models — plus DevOps/Infrastructure tutorials.

## Structure

```
AI_Concepts_Application/
├── .streamlit/config.toml              # Theme & Streamlit config
├── topics/                             # Auto-discovered AI/ML topic modules
│   ├── __init__.py                     # Auto-discovery engine
│   └── learning_path.py               # Starter: Perceptron → LLM roadmap
├── Implementation/                     # Concept implementations (from scratch)
│   └── README.md
├── Automation_Infrastructure/          # Docker, K8s, DevOps tutorials
│   ├── __init__.py                     # Auto-discovery engine
│   ├── _tutorial_template.py           # Template for new tutorials
│   ├── docker_fundamentals.py          # Docker walkthrough
│   └── kubernetes_fundamentals.py      # K8s walkthrough
├── Concept_breakdown/                  # Detailed notes & diagrams
├── Required_Images/                    # Architecture visuals & diagrams
├── app.py                              # Main Streamlit application
├── LLM_module.py                       # AI assistant backend (Anthropic/OpenAI)
├── SolutionGeneration.py               # Vision-based image analysis
├── template.py                         # Template for new Implementation files
├── requirements.txt                    # Python dependencies
├── Keys.env                            # API keys (git-ignored)
├── .gitignore
└── README.md
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key (optional, for AI Assistant)
# Edit Keys.env and add your ANTHROPIC_API_KEY

# 4. Run the app
streamlit run app.py
```

## 4 Main Sections

| Section | What it contains |
|---------|-----------------|
| 📚 **Topics** | AI/ML theory from Perceptron to LLMs (auto-discovered from `topics/`) |
| 🔬 **Implement** | From-scratch implementations with math, code, visualizations (`Implementation/`) |
| 🏗️ **Infra** | Docker, Kubernetes, DevOps tutorials (`Automation_Infrastructure/`) |
| 🤖 **AI Help** | Chat with Claude/GPT about any concept |

## Adding Content

### Topics (AI/ML)
Create a `.py` file in `topics/` with `TOPIC_NAME`, `THEORY`, `COMPLEXITY`, `OPERATIONS`, `get_content()`. Auto-discovered on restart.

### Implementations
Copy `template.py` into `Implementation/`, add `Level:` and `Concepts:` metadata. The template includes 11 sections: overview, intuition, math, architecture, walkthrough, implementation, alternative, pitfalls, connections, demo, and references.

### Infrastructure Tutorials
Copy `Automation_Infrastructure/_tutorial_template.py`, rename without the underscore prefix, fill in `TOPIC_NAME`, `CATEGORY`, `THEORY`, `COMMANDS`, `OPERATIONS`. Auto-discovered on restart.

## AI Assistant
Supports Anthropic Claude, OpenAI GPT, and Mock mode. Add your API key to `Keys.env`.

Create a `Keys.env` file in the project root. The app will auto-detect whichever keys are present and pick the best available provider (Anthropic preferred over OpenAI).

```env
# ─────────────────────────────────────────────────────────────────────────────
# Keys.env  —  API Keys for AI Concepts Reference Hub
# ─────────────────────────────────────────────────────────────────────────────
# Rules:
#   • NO spaces around the = sign
#   • NO quotes around the value
#   • NO trailing spaces after the value
#   • Lines starting with # are comments and are ignored
# ─────────────────────────────────────────────────────────────────────────────

# Anthropic Claude  (preferred LLM provider)
# Get your key at: https://console.anthropic.com
# Format: starts with "sk-ant-api03-", exactly 108 characters
ANTHROPIC_API_KEY=sk-ant-xxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI GPT  (fallback LLM provider)
# Get your key at: https://platform.openai.com/api-keys
# Format: starts with "sk-"
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# HuggingFace  (required for fine-tuning pipeline steps)
# Get your token at: https://huggingface.co/settings/tokens
# Format: starts with "hf_"
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

### Which keys do you need?

| Key                 | Required for                     | Where to get it                                                          |
|---------------------|----------------------------------|--------------------------------------------------------------------------|
| `ANTHROPIC_API_KEY` | AI Help chat (preferred)         | [console.anthropic.com](https://console.anthropic.com)                   |
| `OPENAI_API_KEY`    | AI Help chat (fallback)          | [platform.openai.com/api-keys](https://platform.openai.com/api-keys)     |
| `HF_TOKEN`          | Fine-tuning topic pipeline steps | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

You only need **one** of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for the AI assistant. 
If both are present the app automatically uses Anthropic. `HF_TOKEN` is only needed if you want to run the Full Fine-Tuning, PEFT Additive, or LoRA pipeline steps.

> ⚠️ `Keys.env` is listed in `.gitignore` — it will never be committed to version control.


