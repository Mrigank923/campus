# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Campus market environment client."""

from typing import Dict, Generic, TypeVar

ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")
StateT = TypeVar("StateT")

try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
except ImportError:  # pragma: no cover
    class EnvClient(Generic[ActionT, ObservationT, StateT]):  # type: ignore[no-redef]
        pass

    class StepResult:  # type: ignore[no-redef]
        def __init__(self, observation: object, reward: float | None, done: bool) -> None:
            self.observation = observation
            self.reward = reward
            self.done = done

    class State:  # type: ignore[no-redef]
        def __init__(self, episode_id: str | None, step_count: int = 0) -> None:
            self.episode_id = episode_id
            self.step_count = step_count

from .models import CampusMarketAction, CampusMarketObservation


class CampusEnv(
    EnvClient[CampusMarketAction, CampusMarketObservation, State]
):
    """
    Client for the Campus market environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        if EnvClient is object:
            raise ImportError(
                "openenv is required to use CampusEnv. Install project dependencies first."
            )
        super().__init__(*args, **kwargs)

    def _step_payload(self, action: CampusMarketAction) -> Dict:
        """
        Convert CampusMarketAction to JSON payload for step message.

        Args:
            action: CampusMarketAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {"actions": {agent_id: payload.model_dump() for agent_id, payload in action.actions.items()}}

    def _parse_result(self, payload: Dict) -> StepResult:
        """
        Parse server response into StepResult[CampusMarketObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CampusMarketObservation
        """
        obs_data = payload.get("observation", {})
        observation = CampusMarketObservation(
            episode_id=obs_data.get("episode_id", ""),
            time_step=obs_data.get("time_step", 0),
            max_steps=obs_data.get("max_steps", 0),
            grid_size=obs_data.get("grid_size", {}),
            grid_layout=obs_data.get("grid_layout", []),
            shops=obs_data.get("shops", {}),
            student_clusters=obs_data.get("student_clusters", {}),
            demand_signals=obs_data.get("demand_signals", {}),
            latest_rewards=obs_data.get("latest_rewards", {}),
            latest_step_metrics=obs_data.get("latest_step_metrics", {}),
            info=obs_data.get("info", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
