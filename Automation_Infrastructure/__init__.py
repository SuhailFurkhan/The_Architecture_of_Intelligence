"""
Automation & Infrastructure Tutorials
======================================
This package contains DevOps/Infrastructure tutorial modules
covering Docker, Kubernetes, CI/CD, and related tooling.

Each tutorial module should have:
- TOPIC_NAME: str - Display name for the tutorial
- CATEGORY: str - Category grouping (e.g., "Containers", "Orchestration", "CI/CD")
- THEORY: str - Markdown formatted explanation
- COMMANDS: str - Markdown formatted command reference table
- OPERATIONS: dict - Dictionary of operations with descriptions and code/commands
- get_content(): function - Returns dict with all content

To add a new tutorial:
1. Create a new .py file in this directory
2. Follow the structure of existing modules (see docker_fundamentals.py)
3. The tutorial will be auto-discovered and added to the app

Example structure:
```python
TOPIC_NAME = "Docker Fundamentals"
CATEGORY = "Containers"

THEORY = \"\"\"
### What is Docker?
...
\"\"\"

COMMANDS = \"\"\"
| Command | Description |
|---------|-------------|
\"\"\"

OPERATIONS = {
    "Build an Image": {
        "description": "Build a Docker image from a Dockerfile",
        "code": '''docker build -t myapp:latest .''',
        "language": "bash"
    }
}

def get_content():
    return {
        "theory": THEORY,
        "commands": COMMANDS,
        "operations": OPERATIONS,
        "category": CATEGORY
    }
```
"""

import importlib
import pkgutil
from pathlib import Path


def discover_tutorials():
    """
    Automatically discover and load all tutorial modules.

    Returns:
        dict: {topic_name: topic_content}
    """
    tutorials = {}
    package_path = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name.startswith('_'):
            continue

        try:
            module = importlib.import_module(f'.{module_info.name}', package=__name__)

            if hasattr(module, 'TOPIC_NAME') and hasattr(module, 'get_content'):
                tutorials[module.TOPIC_NAME] = module.get_content()
        except Exception as e:
            print(f"Warning: Could not load tutorial module '{module_info.name}': {e}")

    return tutorials


def get_all_tutorials():
    """Get all tutorials as a dictionary."""
    return discover_tutorials()


# Planned tutorial modules
PLANNED_MODULES = [
    'docker_fundamentals',
    'kubernetes_fundamentals',
    # --- Future ---
    # 'docker_compose',
    # 'dockerfile_best_practices',
    # 'k8s_deployments',
    # 'k8s_services_networking',
    # 'k8s_volumes_storage',
    # 'k8s_helm_charts',
    # 'ci_cd_github_actions',
    # 'ci_cd_jenkins',
    # 'terraform_basics',
    # 'ansible_basics',
    # 'monitoring_prometheus_grafana',
    # 'linux_fundamentals',
    # 'networking_basics',
]
