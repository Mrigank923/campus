from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    class BaseModel:  # type: ignore[no-redef]
        @classmethod
        def _all_annotations(cls) -> dict[str, Any]:
            annotations: dict[str, Any] = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, "__annotations__", {}))
            return annotations

        def __init__(self, **data: Any) -> None:
            annotations = self._all_annotations()
            for name in annotations:
                if name in data:
                    value = data[name]
                else:
                    value = getattr(self.__class__, name, None)
                setattr(self, name, value)

        def model_dump(self) -> dict[str, Any]:
            return {
                key: getattr(self, key)
                for key in self._all_annotations()
            }

    def Field(  # type: ignore[no-redef]
        default: Any = None,
        *,
        default_factory: Any | None = None,
        description: str = "",
    ) -> Any:
        if default_factory is not None:
            return default_factory()
        return default

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:  # pragma: no cover
    class Action(BaseModel):  # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        reward: float | None = None
        done: bool = False
        metadata: dict[str, Any] = Field(default_factory=dict)


class AgentActionPayload(BaseModel):
    business_type: str = Field(..., description="Type of business to operate.")
    price: float = Field(..., description="Price level for the current step.")


class CampusMarketAction(Action):
    actions: dict[str, AgentActionPayload] = Field(
        default_factory=dict,
        description="Per-agent market actions keyed by agent id.",
    )


class CampusMarketObservation(Observation):
    episode_id: str = Field(..., description="Current episode identifier.")
    time_step: int = Field(..., description="Current simulation time step.")
    max_steps: int = Field(..., description="Episode horizon.")
    grid_size: dict[str, int] = Field(
        default_factory=dict,
        description="Grid shape with rows and cols.",
    )
    grid_layout: list[list[dict[str, Any]]] = Field(
        default_factory=list,
        description="Serialized grid cells with shop and student occupancy.",
    )
    shops: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Current shop configuration by agent.",
    )
    student_clusters: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Student cluster distribution and preferences.",
    )
    demand_signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate and local demand indicators.",
    )
    latest_rewards: dict[str, float] = Field(
        default_factory=dict,
        description="Per-agent rewards from the latest step.",
    )
    latest_step_metrics: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Latest per-shop metrics including penalties and profit.",
    )
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed debugging and simulation diagnostics.",
    )
