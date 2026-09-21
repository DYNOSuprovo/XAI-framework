"""Explainer registry.

Built-in explainers register themselves with :func:`register_explainer`; third-party
packages can plug in without touching this repo by exposing an entry point in the
``xai_framework.explainers`` group::

    [project.entry-points."xai_framework.explainers"]
    anchors = "my_package.anchors:AnchorsExplainer"
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .explainers.base import BaseExplainer

ENTRY_POINT_GROUP = "xai_framework.explainers"

_REGISTRY: dict[str, type[BaseExplainer]] = {}
_PLUGINS_LOADED = False


def register_explainer(name: str | None = None):
    """Class decorator that adds an explainer to the registry under ``name``.

    Falls back to the class's ``name`` attribute when no name is given.
    """

    def decorator(cls: type[BaseExplainer]) -> type[BaseExplainer]:
        key = name or getattr(cls, "name", None)
        if not key:
            raise ValueError(f"{cls.__name__} has no `name`; pass one to register_explainer()")
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return decorator


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 API
        eps = entry_points().get(ENTRY_POINT_GROUP, [])
    for ep in eps:
        try:
            cls = ep.load()
        except Exception:  # pragma: no cover - a broken plugin must not break us
            continue
        _REGISTRY.setdefault(ep.name, cls)


def get_explainer(name: str) -> type[BaseExplainer]:
    """Look up an explainer class by name, loading entry-point plugins if needed."""
    if name not in _REGISTRY:
        _load_plugins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown explainer {name!r}; available: {sorted(_REGISTRY)}") from None


def available_explainers() -> dict[str, dict[str, Any]]:
    """Every registered explainer with whether its dependencies are importable."""
    _load_plugins()
    info: dict[str, dict[str, Any]] = {}
    for name, cls in sorted(_REGISTRY.items()):
        ok, reason = cls.is_available()
        info[name] = {
            "class": cls,
            "available": ok,
            "reason": reason,
            "local": cls.supports_local,
            "global": cls.supports_global,
        }
    return info
