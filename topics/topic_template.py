"""
[Topic Name] - [One-line description]
============================================

[Optional: longer overview paragraph you can fill in later]
"""

TOPIC_NAME = "[Topic Name]"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## [Topic Name]

### What is it?
[Definition]

### Why does it matter?
[Motivation]

### How does it work?
[Explanation]
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
# OPERATIONS — Key code snippets for quick reference
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "Example Snippet": {
        "description": "[What this code demonstrates]",
        "runnable": True,
        "code": '''# Your code here
pass
'''
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    return {
        "theory": THEORY,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
    }

