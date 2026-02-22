"""
LLM Module for AI Concepts Reference Hub
==========================================
Provides AI-powered Q&A functionality using various LLM providers.

Supports:
- Anthropic Claude API (with Vision support)
- OpenAI API (optional)
- Local/Mock mode for testing
- Loading API keys from .env files

Usage:
    from LLM_module import LLMAssistant

    assistant = LLMAssistant(provider="anthropic", api_key="your-key")
    response = assistant.query("Explain how attention works in transformers")

    # With image (vision)
    response = assistant.query_with_image(image, "What architecture is shown here?")
"""

import os
import base64
import io
from pathlib import Path
from typing import Optional, Generator, Union
from dataclasses import dataclass
from enum import Enum

# Try to import PIL for image handling
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOCK = "mock"  # For testing without API


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    provider: str
    model: str
    success: bool
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None


def load_env_file(env_path: Union[str, Path] = None) -> dict:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file. If None, searches for Keys.env
                 in the current directory and parent directories.

    Returns:
        Dictionary of environment variables loaded from the file.
    """
    env_vars = {}

    if env_path is None:
        search_paths = [
            Path.cwd() / "Keys.env",
            Path.cwd() / ".env",
            Path(__file__).parent / "Keys.env",
            Path(__file__).parent / ".env",
        ]

        for path in search_paths:
            if path.exists():
                env_path = path
                break

    if env_path is None:
        return env_vars

    env_path = Path(env_path)
    if not env_path.exists():
        return env_vars

    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    if (value.startswith('"') and value.endswith('"')) or \
                            (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]

                    env_vars[key] = value
                    os.environ[key] = value

    except Exception as e:
        print(f"Warning: Could not load env file {env_path}: {e}")

    return env_vars


class LLMAssistant:
    """
    AI Assistant for AI/ML concepts and explanations.

    ┌─────────────────────────────────────────────────────────────────┐
    │                    LLM ASSISTANT FLOW                           │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   User Query ──► System Prompt + Query ──► LLM API              │
    │                                              │                  │
    │                                              ▼                  │
    │                                        Response                 │
    │                                              │                  │
    │                                              ▼                  │
    │                                   LLMResponse Object            │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """

    # System prompt for AI/ML assistance
    SYSTEM_PROMPT = """You are an expert AI/ML research assistant integrated into an AI Concepts Reference Hub application.
Your role is to help users understand:
- Neural network fundamentals (perceptrons, activation functions, backpropagation)
- Deep learning architectures (CNNs, RNNs, LSTMs, Transformers)
- Natural Language Processing and Language Models
- Generative AI (GPT, diffusion models, multimodal models)
- Training techniques (optimization, regularization, fine-tuning)
- Mathematical foundations (linear algebra, calculus, probability for ML)
- Practical implementation in Python (NumPy, PyTorch, TensorFlow)

Guidelines:
1. Provide clear, concise explanations — build intuition before math
2. Include code examples when helpful (use Python + NumPy by default)
3. Explain computational complexity and parameter counts when relevant
4. Use ASCII diagrams to illustrate architectures and data flow
5. Break down complex concepts into digestible steps
6. Connect concepts to their place in the learning path
7. Reference landmark papers when appropriate

Format your responses with:
- Clear headings for different sections
- Code blocks with proper syntax highlighting
- Bullet points for lists
- Mathematical notation in LaTeX where helpful
- Examples and analogies to build intuition"""

    # System prompt for vision/image analysis
    VISION_SYSTEM_PROMPT = """You are an expert AI/ML assistant that can analyze images of neural network architectures, 
equations, training curves, and research paper figures.
When shown an image:
1. Identify the architecture, equation, or concept shown
2. Explain what is depicted in detail
3. Describe the data flow and key components
4. Relate it to the broader AI/ML landscape
5. Provide relevant code if applicable

Always provide clear explanations that connect visuals to underlying concepts."""

    def __init__(
            self,
            provider: str = "anthropic",
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            env_file: Optional[str] = None
    ):
        """
        Initialize the LLM Assistant.

        Args:
            provider: LLM provider ("anthropic", "openai", or "mock")
            api_key: API key for the provider (or set via environment variable)
            model: Specific model to use (uses default if not specified)
            env_file: Path to .env file to load API keys from
        """
        # Always load Keys.env so _get_api_key always has the freshest value
        self._loaded_env = load_env_file(env_file)

        self.provider = LLMProvider(provider.lower())
        self.api_key = api_key
        self.model = model
        self._client = None

        self.default_models = {
            LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
            LLMProvider.OPENAI: "gpt-4o-mini",
            LLMProvider.MOCK: "mock-model"
        }

        if self.model is None:
            self.model = self.default_models.get(self.provider)

    def _get_api_key(self) -> Optional[str]:
        """
        Get API key — Keys.env always takes priority over session state.
        Priority: Keys.env file → system env → passed-in api_key argument.
        """
        env_vars = {
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENAI:    "OPENAI_API_KEY",
        }
        env_var = env_vars.get(self.provider)
        if env_var:
            from_file = self._loaded_env.get(env_var)
            if from_file:
                return from_file
            from_sys = os.environ.get(env_var)
            if from_sys:
                return from_sys
        return self.api_key if self.api_key else None

    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError(
                    "Anthropic API key not provided. Add ANTHROPIC_API_KEY to Keys.env or set it manually.")
            self._client = Anthropic(api_key=api_key)
            return True
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        except Exception as e:
            raise e

    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError("OpenAI API key not provided. Add OPENAI_API_KEY to Keys.env or set it manually.")
            self._client = OpenAI(api_key=api_key)
            return True
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        except Exception as e:
            raise e

    def _image_to_base64(self, image) -> tuple:
        """Convert PIL Image to base64 string."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for image processing. Install with: pip install Pillow")

        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)

        base64_str = base64.standard_b64encode(buffer.read()).decode('utf-8')
        return base64_str, "image/jpeg"

    # ─────────────────────────────────────────────────────────────────
    # ANTHROPIC
    # ─────────────────────────────────────────────────────────────────

    def _query_anthropic(self, user_query: str, context: Optional[str] = None) -> LLMResponse:
        """Query Anthropic Claude API."""
        if self._client is None:
            self._init_anthropic_client()

        messages = []
        if context:
            messages.append({
                "role": "user",
                "content": f"Context from the application:\n{context}\n\nUser question: {user_query}"
            })
        else:
            messages.append({"role": "user", "content": user_query})

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.SYSTEM_PROMPT,
                messages=messages
            )

            content = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

            return LLMResponse(
                content=content, provider="anthropic",
                model=self.model, success=True, tokens_used=tokens
            )
        except Exception as e:
            return LLMResponse(
                content="", provider="anthropic",
                model=self.model, success=False, error_message=str(e)
            )

    def _query_anthropic_with_image(self, image, user_query: str) -> LLMResponse:
        """Query Anthropic Claude API with an image (vision)."""
        if self._client is None:
            self._init_anthropic_client()

        try:
            image_data, media_type = self._image_to_base64(image)

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }
                    },
                    {"type": "text", "text": user_query}
                ]
            }]

            response = self._client.messages.create(
                model=self.model, max_tokens=4096,
                system=self.VISION_SYSTEM_PROMPT, messages=messages
            )

            content = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

            return LLMResponse(
                content=content, provider="anthropic",
                model=self.model, success=True, tokens_used=tokens
            )
        except Exception as e:
            return LLMResponse(
                content="", provider="anthropic",
                model=self.model, success=False, error_message=str(e)
            )

    def _query_anthropic_stream(self, user_query: str, context: Optional[str] = None) -> Generator[str, None, None]:
        """Stream response from Anthropic Claude API."""
        if self._client is None:
            self._init_anthropic_client()

        messages = []
        if context:
            messages.append({
                "role": "user",
                "content": f"Context from the application:\n{context}\n\nUser question: {user_query}"
            })
        else:
            messages.append({"role": "user", "content": user_query})

        try:
            with self._client.messages.stream(
                    model=self.model, max_tokens=4096,
                    system=self.SYSTEM_PROMPT, messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            yield f"\n\n**Error:** {str(e)}"

    # ─────────────────────────────────────────────────────────────────
    # OPENAI
    # ─────────────────────────────────────────────────────────────────

    def _query_openai(self, user_query: str, context: Optional[str] = None) -> LLMResponse:
        """Query OpenAI API."""
        if self._client is None:
            self._init_openai_client()

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if context:
            messages.append({
                "role": "user",
                "content": f"Context from the application:\n{context}\n\nUser question: {user_query}"
            })
        else:
            messages.append({"role": "user", "content": user_query})

        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=4096
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None

            return LLMResponse(
                content=content, provider="openai",
                model=self.model, success=True, tokens_used=tokens
            )
        except Exception as e:
            return LLMResponse(
                content="", provider="openai",
                model=self.model, success=False, error_message=str(e)
            )

    def _query_openai_with_image(self, image, user_query: str) -> LLMResponse:
        """Query OpenAI API with an image (vision)."""
        if self._client is None:
            self._init_openai_client()

        try:
            image_data, media_type = self._image_to_base64(image)

            messages = [
                {"role": "system", "content": self.VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                        {"type": "text", "text": user_query}
                    ]
                }
            ]

            response = self._client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=4096
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else None

            return LLMResponse(
                content=content, provider="openai",
                model="gpt-4o", success=True, tokens_used=tokens
            )
        except Exception as e:
            return LLMResponse(
                content="", provider="openai",
                model="gpt-4o", success=False, error_message=str(e)
            )

    def _query_openai_stream(self, user_query: str, context: Optional[str] = None) -> Generator[str, None, None]:
        """Stream response from OpenAI API."""
        if self._client is None:
            self._init_openai_client()

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if context:
            messages.append({
                "role": "user",
                "content": f"Context from the application:\n{context}\n\nUser question: {user_query}"
            })
        else:
            messages.append({"role": "user", "content": user_query})

        try:
            stream = self._client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=4096, stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n\n**Error:** {str(e)}"

    # ─────────────────────────────────────────────────────────────────
    # MOCK
    # ─────────────────────────────────────────────────────────────────

    def _query_mock(self, user_query: str, context: Optional[str] = None) -> LLMResponse:
        """Return mock response for testing."""
        mock_response = f"""## Mock Response

**Your Question:** {user_query}

This is a mock response for testing purposes. To get real AI responses:

1. Set your API key in the sidebar
2. Choose a provider (Anthropic or OpenAI)
3. Ask your question again

### Example Topics I Can Help With:
- Neural network architectures
- Transformer and attention mechanisms
- Training and optimization techniques
- Generative AI concepts

---
*Mock mode — No API calls made*"""

        return LLMResponse(
            content=mock_response, provider="mock",
            model="mock-model", success=True, tokens_used=0
        )

    def _query_mock_with_image(self, image, user_query: str) -> LLMResponse:
        """Return mock response for image testing."""
        mock_response = f"""## Mock Vision Response

**Your Query:** {user_query}

This is a mock response for testing image analysis. To get real AI analysis:

1. Add your API key to Keys.env (ANTHROPIC_API_KEY=your-key)
2. Choose Anthropic or OpenAI provider
3. The image will be analyzed for AI/ML diagrams and architectures

---
*Mock mode — No API calls made*"""

        return LLMResponse(
            content=mock_response, provider="mock",
            model="mock-model", success=True, tokens_used=0
        )

    def _query_mock_stream(self, user_query: str, context: Optional[str] = None) -> Generator[str, None, None]:
        """Stream mock response for testing."""
        response = self._query_mock(user_query, context)
        words = response.content.split(' ')
        for i, word in enumerate(words):
            yield word + (' ' if i < len(words) - 1 else '')

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def query(self, user_query: str, context: Optional[str] = None) -> LLMResponse:
        """Send a query to the LLM and get a response."""
        if self.provider == LLMProvider.ANTHROPIC:
            return self._query_anthropic(user_query, context)
        elif self.provider == LLMProvider.OPENAI:
            return self._query_openai(user_query, context)
        else:
            return self._query_mock(user_query, context)

    def query_with_image(self, image, user_query: str) -> LLMResponse:
        """Send a query with an image to the LLM (vision capability)."""
        if self.provider == LLMProvider.ANTHROPIC:
            return self._query_anthropic_with_image(image, user_query)
        elif self.provider == LLMProvider.OPENAI:
            return self._query_openai_with_image(image, user_query)
        else:
            return self._query_mock_with_image(image, user_query)

    def query_stream(self, user_query: str, context: Optional[str] = None) -> Generator[str, None, None]:
        """Send a query and stream the response."""
        if self.provider == LLMProvider.ANTHROPIC:
            yield from self._query_anthropic_stream(user_query, context)
        elif self.provider == LLMProvider.OPENAI:
            yield from self._query_openai_stream(user_query, context)
        else:
            yield from self._query_mock_stream(user_query, context)

    def is_configured(self) -> bool:
        """Check if the assistant has valid configuration."""
        if self.provider == LLMProvider.MOCK:
            return True
        return self._get_api_key() is not None

    def get_status(self) -> dict:
        """Get current configuration status."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "configured": self.is_configured(),
            "has_api_key": self._get_api_key() is not None
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


# =============================================================================
# API KEY VALIDATION & AUTO-SELECTION
# =============================================================================

@dataclass
class KeyStatus:
    """Validation result for a single API key."""
    provider:     str
    key_name:     str
    key_preview:  str
    present:      bool
    valid:        bool
    model:        Optional[str]
    hf_username:  Optional[str] = None
    error:        Optional[str] = None


def _make_key_preview(value: str) -> str:
    if not value or len(value) < 12:
        return "***"
    return f"{value[:8]}...{value[-4:]}"


def _validate_anthropic_key(key: str) -> KeyStatus:
    model   = "claude-haiku-4-5-20251001"
    preview = _make_key_preview(key)
    try:
        from anthropic import Anthropic, AuthenticationError, PermissionDeniedError
        client = Anthropic(api_key=key)
        client.messages.create(
            model=model, max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return KeyStatus(provider="anthropic", key_name="ANTHROPIC_API_KEY",
                         key_preview=preview, present=True, valid=True, model=model)
    except Exception as e:
        err = str(e)
        if "rate" in err.lower() or "529" in err:   # rate-limited = key IS valid
            return KeyStatus(provider="anthropic", key_name="ANTHROPIC_API_KEY",
                             key_preview=preview, present=True, valid=True, model=model)
        msg = "401 — Key invalid or revoked" if "401" in err else \
              "403 — Key lacks permissions"  if "403" in err else \
              "anthropic package not installed" if "ImportError" in err else err
        return KeyStatus(provider="anthropic", key_name="ANTHROPIC_API_KEY",
                         key_preview=preview, present=True, valid=False,
                         model=None, error=msg)


def _validate_openai_key(key: str) -> KeyStatus:
    model   = "gpt-4o-mini"
    preview = _make_key_preview(key)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        client.models.list()
        return KeyStatus(provider="openai", key_name="OPENAI_API_KEY",
                         key_preview=preview, present=True, valid=True, model=model)
    except Exception as e:
        err = str(e)
        if "rate" in err.lower():
            return KeyStatus(provider="openai", key_name="OPENAI_API_KEY",
                             key_preview=preview, present=True, valid=True, model=model)
        msg = "401 — Key invalid or revoked" if "401" in err else \
              "openai package not installed"  if "ImportError" in err else err
        return KeyStatus(provider="openai", key_name="OPENAI_API_KEY",
                         key_preview=preview, present=True, valid=False,
                         model=None, error=msg)


def _validate_hf_key(key: str) -> KeyStatus:
    preview = _make_key_preview(key)
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
        return KeyStatus(provider="huggingface", key_name="HF_TOKEN",
                         key_preview=preview, present=True, valid=True,
                         model="HuggingFace Hub", hf_username=data.get("name", "unknown"))
    except Exception as e:
        err = str(e)
        msg = "401 — Token invalid or expired" if "401" in err else err
        return KeyStatus(provider="huggingface", key_name="HF_TOKEN",
                         key_preview=preview, present=True, valid=False,
                         model=None, error=msg)


def _absent(provider: str, key_name: str) -> KeyStatus:
    return KeyStatus(provider=provider, key_name=key_name,
                     key_preview="not set", present=False, valid=False,
                     model=None, error="Key not found in Keys.env")


def validate_all_keys(env_file: str = None) -> dict:
    """
    Validate ANTHROPIC_API_KEY, OPENAI_API_KEY, and HF_TOKEN.
    Returns {key_name: KeyStatus} for all three regardless of presence.
    """
    env_vars = load_env_file(env_file)

    def _get(name):
        return env_vars.get(name) or os.environ.get(name)

    ant_key = _get("ANTHROPIC_API_KEY")
    oai_key = _get("OPENAI_API_KEY")
    hf_key  = _get("HF_TOKEN")

    return {
        "ANTHROPIC_API_KEY": _validate_anthropic_key(ant_key) if ant_key else _absent("anthropic", "ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY":    _validate_openai_key(oai_key)    if oai_key else _absent("openai",    "OPENAI_API_KEY"),
        "HF_TOKEN":          _validate_hf_key(hf_key)         if hf_key  else _absent("huggingface", "HF_TOKEN"),
    }


def auto_select_provider(env_file: str = None) -> dict:
    """
    Live-validate both Anthropic and OpenAI keys and return the best provider.

    Priority:
        1. Anthropic  — preferred when valid
        2. OpenAI     — fallback if Anthropic is invalid/absent
        3. mock       — if neither key is valid

    Returns:
        {
            "provider":  "anthropic" | "openai" | "mock",
            "model":     str | None,
            "reason":    str,
            "anthropic": KeyStatus,
            "openai":    KeyStatus,
        }
    """
    env_vars = load_env_file(env_file)

    def _get(name):
        return env_vars.get(name) or os.environ.get(name)

    ant_key = _get("ANTHROPIC_API_KEY")
    oai_key = _get("OPENAI_API_KEY")

    ant = _validate_anthropic_key(ant_key) if ant_key else _absent("anthropic", "ANTHROPIC_API_KEY")
    oai = _validate_openai_key(oai_key)    if oai_key else _absent("openai",    "OPENAI_API_KEY")

    if ant.valid and oai.valid:
        return {"provider": "anthropic", "model": ant.model,
                "reason": "Both keys valid — using Anthropic (preferred)",
                "anthropic": ant, "openai": oai}
    elif ant.valid:
        return {"provider": "anthropic", "model": ant.model,
                "reason": "Anthropic key valid",
                "anthropic": ant, "openai": oai}
    elif oai.valid:
        return {"provider": "openai", "model": oai.model,
                "reason": "OpenAI key valid (Anthropic unavailable)",
                "anthropic": ant, "openai": oai}
    else:
        return {"provider": "mock", "model": None,
                "reason": "No valid keys found — using mock mode",
                "anthropic": ant, "openai": oai}


def get_available_providers() -> list:
    """Get list of available providers based on installed packages."""
    providers = ["mock"]

    try:
        import anthropic
        providers.append("anthropic")
    except ImportError:
        pass

    try:
        import openai
        providers.append("openai")
    except ImportError:
        pass

    return providers


def format_code_context(topic_name: str = None, operation_name: str = None,
                        code: str = None, implementation_name: str = None) -> str:
    """
    Format context information to send to the LLM.

    Args:
        topic_name: Current topic being viewed
        operation_name: Current operation being viewed
        code: Code snippet being viewed
        implementation_name: Implementation name if applicable

    Returns:
        Formatted context string
    """
    context_parts = []

    if topic_name:
        context_parts.append(f"Current Topic: {topic_name}")

    if operation_name:
        context_parts.append(f"Current Operation: {operation_name}")

    if implementation_name:
        context_parts.append(f"Implementation: {implementation_name}")

    if code:
        context_parts.append(f"Code being viewed:\n```python\n{code}\n```")

    return "\n".join(context_parts) if context_parts else None


def get_api_key_from_env(provider: str = "anthropic", env_file: str = None) -> Optional[str]:
    """Convenience function to get API key from environment file."""
    env_vars = load_env_file(env_file)

    key_names = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY"
    }

    key_name = key_names.get(provider.lower())
    if key_name:
        return env_vars.get(key_name) or os.environ.get(key_name)
    return None


# =============================================================================
# EXAMPLE QUERIES
# =============================================================================

EXAMPLE_QUERIES = [
    "Explain how self-attention works with a step-by-step example",
    "What's the difference between a CNN and an RNN?",
    "How does backpropagation compute gradients through a network?",
    "Explain the Transformer architecture from the ground up",
    "What is the vanishing gradient problem and how do LSTMs solve it?",
    "How does GPT generate text token by token?",
    "Explain RLHF and why it matters for LLM alignment",
    "What is RAG and how does it reduce hallucination?",
]


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing LLM Module with Mock Provider")
    print("=" * 50)

    assistant = LLMAssistant(provider="mock")

    print(f"\nStatus: {assistant.get_status()}")
    print(f"\nAvailable providers: {get_available_providers()}")

    # Test query
    response = assistant.query("What is a Transformer?")
    print(f"\n{response.content}")

    # Test with context
    context = format_code_context(
        topic_name="Transformer Architecture",
        operation_name="Self-Attention",
        code="scores = Q @ K.T / np.sqrt(d_k)"
    )
    print(f"\nContext example:\n{context}")

    # Test env loading
    print(f"\nTesting env file loading...")
    api_key = get_api_key_from_env("anthropic")
    print(f"Anthropic API key found: {bool(api_key)}")