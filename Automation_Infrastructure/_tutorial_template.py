"""
Tutorial Template — Automation & Infrastructure
=================================================
Copy this file, rename it, and fill in the sections.

The TOPIC_NAME and CATEGORY are parsed by the app.
OPERATIONS entries can specify "language" as "bash", "yaml", "python", "json", etc.
"""

TOPIC_NAME = "[Tutorial Name]"
CATEGORY = "[Containers | Orchestration | CI/CD | IaC | Monitoring | Networking]"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## [Tool/Concept Name]

### What is it?
[2-3 sentence definition]

### Why does it exist? (The Problem)
[What pain point does this solve?]

### Key Terminology

| Term              | Definition                                                |
|-------------------|-----------------------------------------------------------|
| [Term 1]          | [Definition]                                              |
| [Term 2]          | [Definition]                                              |

### Architecture Overview

```
    ┌───────────────────────────────────────────┐
    │           [COMPONENT DIAGRAM]              │
    │                                           │
    │   [Component A] ──► [Component B]          │
    │        │                  │                │
    │        ▼                  ▼                │
    │   [Component C]    [Component D]          │
    │                                           │
    └───────────────────────────────────────────┘
```

### How it works (simplified)

1. **Step 1:** [What happens first]
2. **Step 2:** [What happens next]
3. **Step 3:** [Result]

### When to use / When NOT to use

| ✅ Use when                    | ❌ Don't use when                   |
|-------------------------------|-------------------------------------|
| [Scenario 1]                  | [Scenario 1]                        |
| [Scenario 2]                  | [Scenario 2]                        |
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = """
### Quick Reference

| Command                        | What it does                                   |
|--------------------------------|------------------------------------------------|
| `[command 1]`                  | [Description]                                  |
| `[command 2]`                  | [Description]                                  |
| `[command 3]`                  | [Description]                                  |

### Flags & Options Cheatsheet

| Flag          | Long form        | Description                              |
|---------------|------------------|------------------------------------------|
| `-[x]`        | `--[long]`       | [What it does]                           |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS — Step-by-step tutorials with runnable commands
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "[Operation 1 Name]": {
        "description": "[What this operation achieves and when you'd use it]",
        "language": "bash",
        "code": '''# Step 1: [Describe]
[command]

# Step 2: [Describe]
[command]

# Verify
[command]
'''
    },

    "[Operation 2 Name]": {
        "description": "[Description]",
        "language": "yaml",
        "code": '''# [filename].yaml
[yaml content]
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this tutorial module."""
    return {
        "theory": THEORY,
        "commands": COMMANDS,
        "operations": OPERATIONS,
        "category": CATEGORY,
    }
