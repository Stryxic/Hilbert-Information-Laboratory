# hilbert_orchestrator/core/registry.py

from __future__ import annotations
from typing import Dict, List, Callable, Any

from .stages import StageSpec   # OK: imports only dataclasses, no decorators


class StageRegistry:
    def __init__(self) -> None:
        self._stages: Dict[str, StageSpec] = {}

    # ---------------- Registration ----------------

    def register(self, spec: StageSpec) -> None:
        if spec.key in self._stages:
            raise ValueError(f"Duplicate stage key: {spec.key}")
        self._stages[spec.key] = spec

    def decorator(self, **kwargs: Any) -> Callable:
        def wrapper(func: Callable) -> Callable:
            if "key" not in kwargs or "order" not in kwargs or "label" not in kwargs:
                raise ValueError(
                    "Stage registration requires key, order, label."
                )

            spec = StageSpec(func=func, **kwargs)
            self.register(spec)
            return func
        return wrapper

    # ---------------- Accessors -------------------

    def get(self, key: str) -> StageSpec:
        return self._stages[key]

    def all(self) -> List[StageSpec]:
        return list(self._stages.values())

    def get_ordered(self) -> List[StageSpec]:
        return sorted(self._stages.values(), key=lambda s: s.order)

    def clear(self) -> None:
        self._stages.clear()

    def __len__(self) -> int:
        return len(self._stages)

    def __contains__(self, key: str) -> bool:
        return key in self._stages


GLOBAL_STAGE_REGISTRY = StageRegistry()
