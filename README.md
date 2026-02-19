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
