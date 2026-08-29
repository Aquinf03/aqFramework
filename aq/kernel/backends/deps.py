"""Dependency checks for real backends. Fail closed with install hints."""

from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "torch is required for neural / LLM methods. "
            "Install: pip install -r aq/kernel/requirements.txt"
        ) from e
    return torch


def require_transformers():
    require_torch()
    try:
        import transformers
    except ImportError as e:
        raise SystemExit(
            "transformers is required. Install: pip install -r aq/kernel/requirements.txt"
        ) from e
    return transformers


def require_peft():
    require_transformers()
    try:
        import peft
    except ImportError as e:
        raise SystemExit(
            "peft is required for LoRA/QLoRA. Install: pip install -r aq/kernel/requirements.txt"
        ) from e
    return peft


def require_sklearn():
    try:
        import sklearn
    except ImportError as e:
        raise SystemExit(
            "scikit-learn is required for tabular methods. "
            "Install: pip install -r aq/kernel/requirements.txt"
        ) from e
    return sklearn
