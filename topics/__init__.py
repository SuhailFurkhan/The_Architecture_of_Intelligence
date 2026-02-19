"""
Topics Package — AI Concepts Application
=========================================
This package contains all AI/ML reference topics.

Each topic module should have:
- TOPIC_NAME: str - Display name for the topic
- THEORY: str - Markdown formatted theory content
- COMPLEXITY: str - Markdown formatted complexity/comparison table
- OPERATIONS: dict - Dictionary of operations with descriptions and code
- get_content(): function - Returns dict with theory, complexity, operations

To add a new topic:
1. Create a new .py file in this directory (e.g., perceptron.py)
2. Follow the structure of existing modules (see learning_path.py as reference)
3. The topic will be auto-discovered and added to the app

Example structure for a new topic:
```python
TOPIC_NAME = "Perceptron"

THEORY = \"\"\"
### What is a Perceptron?
A perceptron is the simplest neural network unit...
\"\"\"

COMPLEXITY = \"\"\"
| Aspect            | Detail                          |
|-------------------|---------------------------------|
| Parameters        | weights + bias                  |
| Training Time     | O(n * epochs * features)        |
| Inference Time    | O(features)                     |
\"\"\"

OPERATIONS = {
    "Forward Pass": {
        "description": "Compute weighted sum and apply activation",
        "code": '''
def forward(self, x):
    z = np.dot(self.weights, x) + self.bias
    return 1 if z >= 0 else 0
'''
    }
}

def get_content():
    return {
        "theory": THEORY,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS
    }
```
"""

import importlib
import pkgutil
from pathlib import Path


# Auto-discover all topic modules
def discover_topics():
    """
    Automatically discover and load all topic modules.

    Returns:
        dict: {topic_name: topic_content}
    """
    topics = {}
    package_path = Path(__file__).parent

    # Iterate through all .py files in the topics directory
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name.startswith('_'):
            continue  # Skip __init__ and private modules

        try:
            # Import the module
            module = importlib.import_module(f'.{module_info.name}', package=__name__)

            # Check if module has required attributes
            if hasattr(module, 'TOPIC_NAME') and hasattr(module, 'get_content'):
                topics[module.TOPIC_NAME] = module.get_content()
        except Exception as e:
            print(f"Warning: Could not load topic module '{module_info.name}': {e}")

    return topics


# Create a convenient function to get all topics
def get_all_topics():
    """Get all topics as a dictionary."""
    return discover_topics()


# List of planned topic modules (for reference — add as you build them)
PLANNED_MODULES = [
    'learning_path',       # Learning roadmap (starter)
    # --- Foundations ---
    # 'perceptron',
    # 'activation_functions',
    # 'loss_functions',
    # 'gradient_descent',
    # 'backpropagation',
    # --- Neural Networks ---
    # 'feedforward_nn',
    # 'cnn',
    # 'rnn',
    # 'lstm_gru',
    # --- Modern Architectures ---
    # 'attention_mechanism',
    # 'transformer',
    # 'encoder_decoder',
    # --- Generative AI ---
    # 'language_models',
    # 'gpt_architecture',
    # 'fine_tuning',
    # 'rag',
    # 'prompt_engineering',
    # --- Advanced ---
    # 'diffusion_models',
    # 'multimodal_models',
    # 'agents',
]
