from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:  # pragma: no cover
    class Environment:  # type: ignore[no-redef]
        """Fallback base class for local execution without OpenEnv installed."""

    @dataclass
    class State:  # type: ignore[no-redef]
        episode_id: str
        step_count: int

try:
    from ..models import CampusMarketAction, CampusMarketObservation
except ImportError:  # pragma: no cover
    try:
        from models import CampusMarketAction, CampusMarketObservation
    except ImportError:  # pragma: no cover
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.append(str(root))
        try:
            from models import CampusMarketAction, CampusMarketObservation
        except ImportError:  # pragma: no cover
            class CampusMarketAction:  # type: ignore[no-redef]
                def __init__(self, actions: Optional[dict[str, Any]] = None) -> None:
                    self.actions = actions or {}

            class CampusMarketObservation:  # type: ignore[no-redef]
                def __init__(self, **data: Any) -> None:
                    self._data = data

                def model_dump(self) -> dict[str, Any]:
                    return dict(self._data)


BUSINESS_TYPES: tuple[str, ...] = (
    "fast_food",
    "cafe",
    "food",
    "stationery",
    "general_store",
    "fruits_and_juice",
    "tea_stall",
    "medical_store",
    "restaurants",
)


class RandomAdapter:
    """Compatibility wrapper that uses numpy when available, stdlib otherwise."""

    def __init__(self, seed: Optional[int]) -> None:
        self._numpy_rng = np.random.default_rng(seed) if np is not None else None
        self._random = random.Random(seed)

    def choice(
        self,
        population: Sequence[Any] | int,
        size: Optional[int] = None,
        replace: bool = True,
    ) -> Any:
        if self._numpy_rng is not None:
            return self._numpy_rng.choice(population, size=size, replace=replace)

        if isinstance(population, int):
            values = list(range(population))
        else:
            values = list(population)

        if size is None:
            return self._random.choice(values)
        if replace:
            return [self._random.choice(values) for _ in range(size)]
        return self._random.sample(values, size)

    def uniform(
        self, low: float, high: float, size: Optional[int] = None
    ) -> float | list[float]:
        if self._numpy_rng is not None:
            return self._numpy_rng.uniform(low, high, size=size)
        if size is None:
            return self._random.uniform(low, high)
        return [self._random.uniform(low, high) for _ in range(size)]

    def normal(self, mean: float, stddev: float) -> float:
        if self._numpy_rng is not None:
            return float(self._numpy_rng.normal(mean, stddev))
        return self._random.gauss(mean, stddev)

    def integers(self, low: int, high: int) -> int:
        if self._numpy_rng is not None:
            return int(self._numpy_rng.integers(low, high))
        return self._random.randrange(low, high)

    def multinomial(self, n: int, probabilities: Sequence[float]) -> list[int]:
        if self._numpy_rng is not None:
            return self._numpy_rng.multinomial(n, probabilities).tolist()

        counts = [0 for _ in probabilities]
        cumulative: list[float] = []
        running = 0.0
        for probability in probabilities:
            running += probability
            cumulative.append(running)

        for _ in range(n):
            draw = self._random.random()
            for index, threshold in enumerate(cumulative):
                if draw <= threshold:
                    counts[index] += 1
                    break
        return counts


def _sum(values: Sequence[float]) -> float:
    if np is not None:
        return float(np.sum(values))
    return float(sum(values))


def _clip(value: float, low: float, high: float) -> float:
    if np is not None:
        return float(np.clip(value, low, high))
    return float(max(low, min(high, value)))


@dataclass(frozen=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class PriceRange:
    min_price: float
    max_price: float
    default_price: float


@dataclass
class EnvironmentConfig:
    grid_size: tuple[int, int] = (10, 10)
    num_agents: int = 4
    num_student_clusters: int = 8
    max_steps: int = 50
    min_price: float = 1.0
    max_price: float = 10.0
    initial_price: float = 5.0
    price_ranges_by_type: dict[str, PriceRange] | None = None
    base_operating_cost: float = 8.0
    variable_cost_ratio: float = 0.35
    competition_radius: float = 3.5
    competition_penalty_weight: float = 3.0
    demand_mismatch_penalty_weight: float = 5.0
    distance_weight: float = 0.55
    preference_weight: float = 2.8
    price_weight: float = 1.7
    budget_weight: float = 1.0
    outside_option_bias: float = 0.35
    demand_noise_scale: float = 0.15
    minimum_population: int = 18
    maximum_population: int = 65
    minimum_budget: float = 3.0
    maximum_budget: float = 12.0
    minimum_price_sensitivity: float = 0.5
    maximum_price_sensitivity: float = 2.0


@dataclass
class ShopAgent:
    agent_id: str
    position: Position
    business_type: str
    price: float


@dataclass
class StudentCluster:
    cluster_id: str
    position: Position
    population: int
    budget: float
    preferences: dict[str, float]
    price_sensitivity: float


@dataclass
class ShopStepMetrics:
    customers: int = 0
    revenue: float = 0.0
    operating_cost: float = 0.0
    profit: float = 0.0
    competition_penalty: float = 0.0
    demand_mismatch_penalty: float = 0.0
    reward: float = 0.0


@dataclass
class StepComputation:
    rewards: dict[str, float]
    info: dict[str, Any]
    metrics: dict[str, ShopStepMetrics]


class CampusMarketEnv(Environment):
    """Multi-agent campus market simulation environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        seed: Optional[int] = 7,
        grid_size: tuple[int, int] = (10, 10),
        num_agents: int = 4,
        num_student_clusters: int = 8,
        max_steps: int = 50,
        min_price: float = 1.0,
        max_price: float = 10.0,
        competition_radius: float = 3.5,
        demand_noise_scale: float = 0.15,
    ) -> None:
        fallback_price_ranges = self._build_default_price_ranges(min_price, max_price)
        self.config = EnvironmentConfig(
            grid_size=grid_size,
            num_agents=num_agents,
            num_student_clusters=num_student_clusters,
            max_steps=max_steps,
            min_price=min_price,
            max_price=max_price,
            initial_price=(min_price + max_price) / 2.0,
            price_ranges_by_type=fallback_price_ranges,
            competition_radius=competition_radius,
            demand_noise_scale=demand_noise_scale,
        )
        self.seed = seed
        self._rng = RandomAdapter(seed)
        self._state_meta = State(episode_id=str(uuid4()), step_count=0)
        self._shops: dict[str, ShopAgent] = {}
        self._student_clusters: list[StudentCluster] = []
        self._latest_info: dict[str, Any] = {}
        self._latest_rewards: dict[str, float] = {}
        self._last_metrics: dict[str, ShopStepMetrics] = {}
        self._done = False
        self._cached_state: dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        self._rng = RandomAdapter(self.seed)
        self._state_meta = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._latest_info = {}
        self._latest_rewards = {agent_id: 0.0 for agent_id in self._agent_ids()}
        self._last_metrics = {}
        self._initialize_entities()
        self._cached_state = self._build_state()
        return self._cached_state

    def step(
        self,
        action: Mapping[str, Any] | CampusMarketAction,
    ) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]] | dict[str, Any]:
        if not self._shops:
            self.reset()

        if self._done:
            return self._format_step_result(
                self._cached_state,
                self._latest_rewards,
                True,
                {
                    **self._latest_info,
                    "warning": "Episode already terminated. Call reset() to start a new episode.",
                },
                openenv=isinstance(action, CampusMarketAction),
            )

        action_dict = self._unwrap_actions(action)
        normalized_actions = self._normalize_actions(action_dict)
        self._apply_actions(normalized_actions)

        self._state_meta.step_count += 1
        computation = self._simulate_step(normalized_actions)
        self._latest_rewards = computation.rewards
        self._latest_info = computation.info
        self._last_metrics = computation.metrics
        self._done = self._state_meta.step_count >= self.config.max_steps
        self._cached_state = self._build_state()

        return self._format_step_result(
            self._cached_state,
            computation.rewards,
            self._done,
            computation.info,
            openenv=isinstance(action, CampusMarketAction),
        )

    def state(self) -> dict[str, Any]:
        if not self._cached_state:
            self._cached_state = self.reset()
        return self._cached_state

    def sample_random_actions(self) -> dict[str, dict[str, Any]]:
        actions: dict[str, dict[str, Any]] = {}
        for agent_id in self._shops or {agent_id: None for agent_id in self._agent_ids()}:
            business_type = str(self._rng.choice(BUSINESS_TYPES))
            price_range = self._price_range_for_type(business_type)
            actions[agent_id] = {
                "business_type": business_type,
                "price": float(
                    self._rng.uniform(price_range.min_price, price_range.max_price)
                ),
            }
        return actions

    def _agent_ids(self) -> list[str]:
        return [f"agent_{index}" for index in range(self.config.num_agents)]

    def _initialize_entities(self) -> None:
        rows, cols = self.config.grid_size
        occupied = self._sample_unique_positions(
            self.config.num_agents + self.config.num_student_clusters
        )
        shop_positions = occupied[: self.config.num_agents]
        cluster_positions = occupied[self.config.num_agents :]

        self._shops = {}
        for index, position in enumerate(shop_positions):
            agent_id = f"agent_{index}"
            self._shops[agent_id] = ShopAgent(
                agent_id=agent_id,
                position=position,
                business_type=BUSINESS_TYPES[index % len(BUSINESS_TYPES)],
                price=float(
                    self._price_range_for_type(
                        BUSINESS_TYPES[index % len(BUSINESS_TYPES)]
                    ).default_price
                ),
            )

        self._student_clusters = []
        for index, position in enumerate(cluster_positions):
            raw_preferences = self._rng.uniform(0.1, 1.0, size=len(BUSINESS_TYPES))
            preference_total = _sum(raw_preferences)
            preferences = {
                business_type: float(raw_preferences[type_index] / preference_total)
                for type_index, business_type in enumerate(BUSINESS_TYPES)
            }
            self._student_clusters.append(
                StudentCluster(
                    cluster_id=f"cluster_{index}",
                    position=position,
                    population=int(
                        self._rng.integers(
                            self.config.minimum_population,
                            self.config.maximum_population + 1,
                        )
                    ),
                    budget=float(
                        self._rng.uniform(
                            self.config.minimum_budget, self.config.maximum_budget
                        )
                    ),
                    preferences=preferences,
                    price_sensitivity=float(
                        self._rng.uniform(
                            self.config.minimum_price_sensitivity,
                            self.config.maximum_price_sensitivity,
                        )
                    ),
                )
            )

        self._latest_rewards = {agent_id: 0.0 for agent_id in self._shops}
        self._last_metrics = {
            agent_id: ShopStepMetrics() for agent_id in self._shops
        }

    def _sample_unique_positions(self, count: int) -> list[Position]:
        rows, cols = self.config.grid_size
        total_cells = rows * cols
        if count > total_cells:
            raise ValueError(
                f"Grid {self.config.grid_size} cannot fit {count} unique entities."
            )
        flat_indices = self._rng.choice(total_cells, size=count, replace=False)
        positions: list[Position] = []
        indices = flat_indices.tolist() if hasattr(flat_indices, "tolist") else list(flat_indices)
        for index in indices:
            x, y = divmod(index, cols)
            positions.append(Position(x=x, y=y))
        return positions

    def _unwrap_actions(
        self, action: Mapping[str, Any] | CampusMarketAction
    ) -> Mapping[str, Any]:
        if isinstance(action, CampusMarketAction):
            return action.actions
        return action

    def _normalize_actions(
        self, action_dict: Mapping[str, Any]
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for agent_id, shop in self._shops.items():
            raw_action = action_dict.get(agent_id, {}) if isinstance(action_dict, Mapping) else {}
            if hasattr(raw_action, "model_dump"):
                payload: Mapping[str, Any] = raw_action.model_dump()
            elif isinstance(raw_action, Mapping):
                payload = raw_action
            else:
                payload = {}

            business_type = str(payload.get("business_type", shop.business_type)).lower()
            if business_type not in BUSINESS_TYPES:
                business_type = shop.business_type
            price_range = self._price_range_for_type(business_type)

            raw_price = payload.get("price", shop.price)
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                price = shop.price

            price = _clip(price, price_range.min_price, price_range.max_price)
            normalized[agent_id] = {
                "business_type": business_type,
                "price": round(price, 4),
                "price_range": self._price_range_dict(price_range),
            }
        return normalized

    def _apply_actions(self, normalized_actions: Mapping[str, Mapping[str, Any]]) -> None:
        for agent_id, payload in normalized_actions.items():
            shop = self._shops[agent_id]
            shop.business_type = str(payload["business_type"])
            shop.price = float(payload["price"])

    def _simulate_step(
        self, normalized_actions: Mapping[str, Mapping[str, Any]]
    ) -> StepComputation:
        metrics = {agent_id: ShopStepMetrics() for agent_id in self._shops}
        cluster_logs: list[dict[str, Any]] = []

        demand_by_type = {business_type: 0.0 for business_type in BUSINESS_TYPES}
        demand_near_shop = {agent_id: 0.0 for agent_id in self._shops}
        competition_scores = self._competition_scores()

        for cluster in self._student_clusters:
            noise = _clip(self._rng.normal(1.0, self.config.demand_noise_scale), 0.5, 1.5)
            active_population = max(0, int(round(cluster.population * noise)))
            if active_population == 0:
                continue

            for business_type in BUSINESS_TYPES:
                demand_by_type[business_type] += (
                    active_population * cluster.preferences[business_type]
                )

            scores: dict[str, float] = {}
            score_details: dict[str, dict[str, float]] = {}
            for agent_id, shop in self._shops.items():
                score, detail = self._score_shop(cluster, shop)
                scores[agent_id] = score
                score_details[agent_id] = detail
                demand_near_shop[agent_id] += (
                    active_population * cluster.preferences[shop.business_type]
                )

            allocations, abstained = self._allocate_customers(active_population, scores)
            purchased = active_population - abstained

            for agent_id, customers in allocations.items():
                if customers <= 0:
                    continue
                shop = self._shops[agent_id]
                shop_metrics = metrics[agent_id]
                shop_metrics.customers += customers
                shop_metrics.revenue += shop.price * customers

            cluster_logs.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "position": self._position_dict(cluster.position),
                    "budget": round(cluster.budget, 4),
                    "price_sensitivity": round(cluster.price_sensitivity, 4),
                    "active_population": active_population,
                    "purchased": purchased,
                    "abstained": abstained,
                    "preferences": {
                        key: round(value, 4)
                        for key, value in cluster.preferences.items()
                    },
                    "shop_scores": {
                        agent_id: {
                            "score": round(scores[agent_id], 4),
                            **{
                                key: round(value, 4)
                                for key, value in score_details[agent_id].items()
                            },
                            "allocated_customers": allocations[agent_id],
                        }
                        for agent_id in self._shops
                    },
                }
            )

        rewards: dict[str, float] = {}
        per_shop_info: dict[str, Any] = {}
        for agent_id, shop in self._shops.items():
            shop_metrics = metrics[agent_id]
            shop_metrics.operating_cost = (
                self.config.base_operating_cost
                + self.config.variable_cost_ratio * shop.price * shop_metrics.customers
            )
            shop_metrics.profit = shop_metrics.revenue - shop_metrics.operating_cost

            expected_local_demand = max(demand_near_shop[agent_id], 1.0)
            realized_customers = float(shop_metrics.customers)
            mismatch_ratio = max(
                0.0, (expected_local_demand - realized_customers) / expected_local_demand
            )
            shop_metrics.demand_mismatch_penalty = (
                mismatch_ratio * self.config.demand_mismatch_penalty_weight
            )
            shop_metrics.competition_penalty = (
                competition_scores[agent_id] * self.config.competition_penalty_weight
            )
            shop_metrics.reward = (
                shop_metrics.profit
                - shop_metrics.competition_penalty
                - shop_metrics.demand_mismatch_penalty
            )
            rewards[agent_id] = round(shop_metrics.reward, 4)
            per_shop_info[agent_id] = {
                "business_type": shop.business_type,
                "price": round(shop.price, 4),
                "price_range": self._price_range_dict(
                    self._price_range_for_type(shop.business_type)
                ),
                "position": self._position_dict(shop.position),
                "customers": shop_metrics.customers,
                "revenue": round(shop_metrics.revenue, 4),
                "operating_cost": round(shop_metrics.operating_cost, 4),
                "profit": round(shop_metrics.profit, 4),
                "competition_penalty": round(shop_metrics.competition_penalty, 4),
                "demand_mismatch_penalty": round(
                    shop_metrics.demand_mismatch_penalty, 4
                ),
                "reward": rewards[agent_id],
                "competition_score": round(competition_scores[agent_id], 4),
                "local_expected_demand": round(expected_local_demand, 4),
            }

        info = {
            "step": self._state_meta.step_count,
            "episode_id": self._state_meta.episode_id,
            "normalized_actions": normalized_actions,
            "demand_by_type": {
                business_type: round(value, 4)
                for business_type, value in demand_by_type.items()
            },
            "per_shop": per_shop_info,
            "cluster_choices": cluster_logs,
            "competition_scores": {
                agent_id: round(score, 4)
                for agent_id, score in competition_scores.items()
            },
        }
        return StepComputation(rewards=rewards, info=info, metrics=metrics)

    def _score_shop(
        self, cluster: StudentCluster, shop: ShopAgent
    ) -> tuple[float, dict[str, float]]:
        price_range = self._price_range_for_type(shop.business_type)
        distance = self._distance(cluster.position, shop.position)
        distance_term = math.exp(-self.config.distance_weight * distance)
        preference_term = cluster.preferences[shop.business_type]
        affordability_gap = max(0.0, shop.price - cluster.budget)
        budget_term = math.exp(-self.config.budget_weight * affordability_gap)
        price_term = math.exp(
            -self.config.price_weight
            * cluster.price_sensitivity
            * (
                (shop.price - price_range.min_price)
                / max(price_range.max_price - price_range.min_price, 1e-6)
            )
        )
        score = (
            self.config.preference_weight * preference_term
            + distance_term
            + budget_term
            + price_term
        )
        score = max(score, 1e-6)
        return score, {
            "distance": distance,
            "distance_term": distance_term,
            "preference_term": preference_term,
            "budget_term": budget_term,
            "price_term": price_term,
            "price_range_min": price_range.min_price,
            "price_range_max": price_range.max_price,
        }

    def _allocate_customers(
        self, active_population: int, scores: Mapping[str, float]
    ) -> tuple[dict[str, int], int]:
        if active_population <= 0:
            return {agent_id: 0 for agent_id in self._shops}, 0

        shop_agent_ids = list(self._shops.keys())
        shop_scores = [float(scores[agent_id]) for agent_id in shop_agent_ids]
        outside_score = float(self.config.outside_option_bias)
        total = _sum(shop_scores) + outside_score
        probabilities = [score / total for score in shop_scores]
        probabilities.append(outside_score / total)
        draws = self._rng.multinomial(active_population, probabilities)

        allocations = {
            agent_id: int(draws[index]) for index, agent_id in enumerate(shop_agent_ids)
        }
        abstained = int(draws[-1])
        return allocations, abstained

    def _competition_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for agent_id, shop in self._shops.items():
            score = 0.0
            for other_agent_id, other_shop in self._shops.items():
                if other_agent_id == agent_id:
                    continue
                if other_shop.business_type != shop.business_type:
                    continue
                distance = self._distance(shop.position, other_shop.position)
                if distance > self.config.competition_radius:
                    continue
                proximity = 1.0 - (distance / max(self.config.competition_radius, 1e-6))
                score += max(proximity, 0.0)
            scores[agent_id] = score
        return scores

    def _build_state(self) -> dict[str, Any]:
        rows, cols = self.config.grid_size
        grid_layout: list[list[dict[str, Any]]] = []
        for x in range(rows):
            row: list[dict[str, Any]] = []
            for y in range(cols):
                shops_here = [
                    agent_id
                    for agent_id, shop in self._shops.items()
                    if shop.position.x == x and shop.position.y == y
                ]
                clusters_here = [
                    cluster.cluster_id
                    for cluster in self._student_clusters
                    if cluster.position.x == x and cluster.position.y == y
                ]
                if shops_here:
                    cell_type = "shop"
                elif clusters_here:
                    cell_type = "student_cluster"
                else:
                    cell_type = "empty"
                row.append(
                    {
                        "x": x,
                        "y": y,
                        "type": cell_type,
                        "shops": shops_here,
                        "student_clusters": clusters_here,
                    }
                )
            grid_layout.append(row)

        demand_signals = self._demand_signals()
        latest_step = {
            agent_id: {
                "customers": metrics.customers,
                "revenue": round(metrics.revenue, 4),
                "profit": round(metrics.profit, 4),
                "competition_penalty": round(metrics.competition_penalty, 4),
                "demand_mismatch_penalty": round(metrics.demand_mismatch_penalty, 4),
                "reward": round(metrics.reward, 4),
            }
            for agent_id, metrics in self._last_metrics.items()
        }

        return {
            "episode_id": self._state_meta.episode_id,
            "time_step": self._state_meta.step_count,
            "max_steps": self.config.max_steps,
            "done": self._done,
            "business_types": list(BUSINESS_TYPES),
            "price_ranges_by_type": self._serialize_price_ranges(),
            "grid_size": {"rows": rows, "cols": cols},
            "grid_layout": grid_layout,
            "shops": {
                agent_id: {
                    "agent_id": shop.agent_id,
                    "position": self._position_dict(shop.position),
                    "business_type": shop.business_type,
                    "price": round(shop.price, 4),
                    "price_range": self._price_range_dict(
                        self._price_range_for_type(shop.business_type)
                    ),
                }
                for agent_id, shop in self._shops.items()
            },
            "student_clusters": {
                cluster.cluster_id: {
                    "cluster_id": cluster.cluster_id,
                    "position": self._position_dict(cluster.position),
                    "population": cluster.population,
                    "budget": round(cluster.budget, 4),
                    "preferences": {
                        key: round(value, 4)
                        for key, value in cluster.preferences.items()
                    },
                    "price_sensitivity": round(cluster.price_sensitivity, 4),
                }
                for cluster in self._student_clusters
            },
            "demand_signals": demand_signals,
            "latest_rewards": self._latest_rewards,
            "latest_step_metrics": latest_step,
        }

    def _demand_signals(self) -> dict[str, Any]:
        aggregate = {business_type: 0.0 for business_type in BUSINESS_TYPES}
        local = {agent_id: 0.0 for agent_id in self._shops}
        for cluster in self._student_clusters:
            for business_type in BUSINESS_TYPES:
                aggregate[business_type] += (
                    cluster.population * cluster.preferences[business_type]
                )
            for agent_id, shop in self._shops.items():
                distance = self._distance(cluster.position, shop.position)
                proximity = math.exp(-0.35 * distance)
                local[agent_id] += (
                    cluster.population
                    * cluster.preferences[shop.business_type]
                    * proximity
                )
        return {
            "aggregate_by_type": {
                business_type: round(value, 4)
                for business_type, value in aggregate.items()
            },
            "local_by_shop": {
                agent_id: round(value, 4) for agent_id, value in local.items()
            },
        }

    def _format_step_result(
        self,
        next_state: dict[str, Any],
        rewards: dict[str, float],
        done: bool,
        info: dict[str, Any],
        *,
        openenv: bool,
    ) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]] | dict[str, Any]:
        if openenv:
            total_reward = float(sum(rewards.values()))
            observation = CampusMarketObservation(
                episode_id=next_state["episode_id"],
                time_step=next_state["time_step"],
                max_steps=next_state["max_steps"],
                done=done,
                grid_size=next_state["grid_size"],
                grid_layout=next_state["grid_layout"],
                shops=next_state["shops"],
                student_clusters=next_state["student_clusters"],
                demand_signals=next_state["demand_signals"],
                latest_rewards=rewards,
                latest_step_metrics=next_state["latest_step_metrics"],
                info=info,
                reward=total_reward,
                metadata={"per_agent_rewards": rewards, "info": info},
            )
            return observation.model_dump()
        return next_state, rewards, done, info

    @staticmethod
    def _distance(a: Position, b: Position) -> float:
        return math.dist((a.x, a.y), (b.x, b.y))

    @staticmethod
    def _position_dict(position: Position) -> dict[str, int]:
        return asdict(position)

    @staticmethod
    def _build_default_price_ranges(
        global_min_price: float, global_max_price: float
    ) -> dict[str, PriceRange]:
        def scaled(low_ratio: float, high_ratio: float, default_ratio: float) -> PriceRange:
            min_value = global_min_price + (global_max_price - global_min_price) * low_ratio
            max_value = global_min_price + (global_max_price - global_min_price) * high_ratio
            default_value = global_min_price + (global_max_price - global_min_price) * default_ratio
            return PriceRange(
                min_price=round(min_value, 4),
                max_price=round(max_value, 4),
                default_price=round(default_value, 4),
            )

        return {
            "tea_stall": scaled(0.0, 0.22, 0.08),
            "fast_food": scaled(0.08, 0.4, 0.2),
            "fruits_and_juice": scaled(0.1, 0.48, 0.24),
            "stationery": scaled(0.08, 0.5, 0.22),
            "food": scaled(0.18, 0.62, 0.34),
            "general_store": scaled(0.14, 0.68, 0.36),
            "cafe": scaled(0.28, 0.78, 0.5),
            "medical_store": scaled(0.18, 0.88, 0.46),
            "restaurants": scaled(0.42, 1.0, 0.68),
        }

    def _price_range_for_type(self, business_type: str) -> PriceRange:
        price_ranges = self.config.price_ranges_by_type or {}
        if business_type in price_ranges:
            return price_ranges[business_type]
        return PriceRange(
            min_price=self.config.min_price,
            max_price=self.config.max_price,
            default_price=self.config.initial_price,
        )

    @staticmethod
    def _price_range_dict(price_range: PriceRange) -> dict[str, float]:
        return {
            "min_price": round(price_range.min_price, 4),
            "max_price": round(price_range.max_price, 4),
            "default_price": round(price_range.default_price, 4),
        }

    def _serialize_price_ranges(self) -> dict[str, dict[str, float]]:
        price_ranges = self.config.price_ranges_by_type or {}
        return {
            business_type: self._price_range_dict(price_range)
            for business_type, price_range in price_ranges.items()
        }


def sample_random_actions(
    env: Optional[CampusMarketEnv] = None,
    seed: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    active_env = env or CampusMarketEnv(seed=seed)
    if not active_env._shops:
        active_env.reset(seed=seed)
    return active_env.sample_random_actions()


if __name__ == "__main__":
    environment = CampusMarketEnv(seed=21)
    state = environment.reset()
    print("Initial state keys:", sorted(state.keys()))
    for _ in range(5):
        actions = sample_random_actions(environment)
        state, rewards, done, info = environment.step(actions)
        print(rewards)
        if done:
            break
