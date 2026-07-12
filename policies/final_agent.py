"""Heuristic planning agent for the Overcooked final project.

The runner loads this class through StudentAgentAdapter. It expects observations
with observation.type: state, because the planner needs raw Overcooked state and
MDP objects.
"""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from typing import Iterable

import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import Recipe


ACTION_TO_INDEX = {
    Direction.NORTH: 0,
    Direction.SOUTH: 1,
    Direction.EAST: 2,
    Direction.WEST: 3,
    Action.STAY: 4,
    Action.INTERACT: 5,
}


class StudentAgent:
    """Fast symbolic agent: task planner + A* navigation.

    Priority is intentionally soup-centric:
    1. Deliver soup already held.
    2. Use held dish on ready soup.
    3. Put held ingredient into a pot that can accept it.
    4. Empty-handed: get dish when soup is ready/cooking soon.
    5. Empty-handed: bring ingredients to pots.
    6. Otherwise wait near a useful feature without blocking the teammate.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.ingredient = str(self.config.get("ingredient", "onion")).lower()
        self.avoid_teammate = bool(self.config.get("avoid_teammate", True))
        self.teammate_aware = bool(self.config.get("teammate_aware", True))
        self.rng = np.random.default_rng(self.config.get("seed", None))
        self._layout_name = None
        self._valid_positions: set[tuple[int, int]] = set()
        self._distance_cache: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    def reset(self):
        self._layout_name = None
        self._valid_positions = set()
        self._distance_cache = {}

    def act(self, obs):
        try:
            state = obs["state"]
            mdp = obs["mdp"]
            agent_index = int(obs.get("agent_index", 0))
            self._refresh_layout_cache(mdp)

            target = self._choose_target(state, mdp, agent_index)
            if target is None:
                return 4

            action = self._move_or_interact(state, mdp, agent_index, target)
            return int(ACTION_TO_INDEX.get(action, 4))
        except Exception:
            return 4

    # ------------------------------------------------------------------
    # Task planning
    # ------------------------------------------------------------------

    def _choose_target(self, state, mdp, agent_index: int) -> tuple[int, int] | None:
        player = state.players[agent_index]
        held = player.held_object
        pot_states = mdp.get_pot_states(state)

        if held is not None:
            held_name = held.name
            if held_name == "soup":
                return self._best_feature_target(player.position, mdp.get_serving_locations())

            if held_name == "dish":
                ready = list(pot_states.get("ready", []))
                if ready:
                    return self._best_feature_target(player.position, ready)
                nearly_ready = list(pot_states.get("cooking", [])) + list(
                    pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", [])
                )
                return self._best_feature_target(player.position, nearly_ready)

            if held_name in {"onion", "tomato"}:
                pots = self._pots_accepting_ingredients(pot_states)
                if not pots:
                    return self._best_feature_target(player.position, mdp.get_empty_counter_locations(state))
                if self._teammate_can_feed_pot(state, mdp, agent_index, pot_states):
                    parking = self._best_floor_target(player.position, self._parking_positions(state, mdp, agent_index))
                    if parking is not None:
                        return parking
                return self._best_feature_target(player.position, pots)

            return None

        return self._choose_empty_handed_target(state, mdp, agent_index, pot_states)

    def _choose_empty_handed_target(self, state, mdp, agent_index: int, pot_states) -> tuple[int, int] | None:
        player = state.players[agent_index]
        pos = player.position
        teammate = state.players[1 - agent_index]
        teammate_held = None if teammate.held_object is None else teammate.held_object.name

        candidates: list[tuple[float, tuple[int, int]]] = []
        ready = list(pot_states.get("ready", []))
        cooking = list(pot_states.get("cooking", []))
        full = list(pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", []))
        accepting = self._pots_accepting_ingredients(pot_states)

        if ready:
            for dish in self._dish_sources(state, mdp):
                candidates.append((1000 - self._feature_distance(pos, dish), dish))

            # If a dish is already handled by teammate, start the next soup cycle.
            if self.teammate_aware and teammate_held == "dish":
                for ingredient in self._ingredient_sources(state, mdp):
                    candidates.append((780 - self._feature_distance(pos, ingredient), ingredient))

        if full:
            for pot in full:
                candidates.append((900 - self._feature_distance(pos, pot), pot))

        if cooking:
            for dish in self._dish_sources(state, mdp):
                candidates.append((820 - self._feature_distance(pos, dish), dish))

        if accepting:
            ingredient_priority = 760
            if self.teammate_aware and teammate_held in {"onion", "tomato"}:
                ingredient_priority -= 220
            for ingredient in self._ingredient_sources(state, mdp):
                candidates.append((ingredient_priority - self._feature_distance(pos, ingredient), ingredient))

        if not candidates:
            idle_targets = list(cooking) + list(ready) + list(mdp.get_dish_dispenser_locations())
            return self._best_feature_target(pos, idle_targets)

        return max(candidates, key=lambda item: item[0])[1]

    def _pots_accepting_ingredients(self, pot_states) -> list[tuple[int, int]]:
        pots: list[tuple[int, int]] = []
        pots.extend(list(pot_states.get("empty", [])))
        for n_items in range(1, Recipe.MAX_NUM_INGREDIENTS):
            pots.extend(list(pot_states.get(f"{n_items}_items", [])))
        return pots

    def _teammate_can_feed_pot(self, state, mdp, agent_index: int, pot_states) -> bool:
        if not self.teammate_aware:
            return False
        teammate = state.players[1 - agent_index]
        teammate_held = None if teammate.held_object is None else teammate.held_object.name
        if teammate_held not in {"onion", "tomato"}:
            return False
        accepting = self._pots_accepting_ingredients(pot_states)
        if not accepting:
            return False
        teammate_dist = min(self._feature_distance(teammate.position, pot) for pot in accepting)
        return teammate_dist <= 1

    def _parking_positions(self, state, mdp, agent_index: int) -> list[tuple[int, int]]:
        player = state.players[agent_index]
        teammate = state.players[1 - agent_index]
        critical_features = (
            list(mdp.get_dish_dispenser_locations())
            + self._ingredient_dispenser_locations(mdp)
            + list(mdp.get_serving_locations())
            + list(mdp.get_pot_locations())
        )
        blocked_access = {
            pos
            for feature in critical_features
            for pos in self._adjacent_positions(feature)
            if pos in self._valid_positions
        }
        strict = [
            pos
            for pos in self._valid_positions
            if pos not in blocked_access and pos != teammate.position and pos != player.position
        ]
        if strict:
            return strict

        teammate_front = Action.move_in_direction(teammate.position, teammate.orientation)
        return [
            pos
            for pos in self._valid_positions
            if pos != teammate.position and pos != player.position and pos != teammate_front
        ]

    def _dish_sources(self, state, mdp) -> list[tuple[int, int]]:
        return self._counter_objects_by_name(state, "dish") + list(mdp.get_dish_dispenser_locations())

    def _ingredient_sources(self, state, mdp) -> list[tuple[int, int]]:
        counter_items = self._counter_objects_by_name(state, self.ingredient)
        dispenser_items = self._ingredient_dispenser_locations(mdp)
        if counter_items:
            return counter_items + dispenser_items
        return dispenser_items

    def _ingredient_dispenser_locations(self, mdp) -> list[tuple[int, int]]:
        if self.ingredient == "tomato":
            locs = list(mdp.get_tomato_dispenser_locations())
            if locs:
                return locs
        onion_locs = list(mdp.get_onion_dispenser_locations())
        if onion_locs:
            return onion_locs
        return list(mdp.get_tomato_dispenser_locations())

    @staticmethod
    def _counter_objects_by_name(state, object_name: str) -> list[tuple[int, int]]:
        return [obj.position for obj in state.objects.values() if obj.name == object_name]

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _move_or_interact(self, state, mdp, agent_index: int, target: tuple[int, int]):
        player = state.players[agent_index]
        pos = player.position

        if target in self._valid_positions:
            if pos == target:
                return Action.STAY
            next_pos = self._next_step_to_floor(state, agent_index, target)
            if next_pos is None:
                return self._unstuck_action(state, mdp, agent_index)
            return Action.determine_action_for_change_in_pos(pos, next_pos)

        if self._is_adjacent(pos, target):
            direction = self._direction_from_to(pos, target)
            if player.orientation == direction:
                return Action.INTERACT
            return direction

        next_pos = self._next_step_to_feature(state, mdp, agent_index, target)
        if next_pos is None:
            return self._unstuck_action(state, mdp, agent_index)
        return Action.determine_action_for_change_in_pos(pos, next_pos)

    def _next_step_to_feature(self, state, mdp, agent_index: int, target: tuple[int, int]) -> tuple[int, int] | None:
        player = state.players[agent_index]
        start = player.position
        blocked = self._blocked_positions(state, agent_index)
        goals = [
            pos
            for pos in self._adjacent_positions(target)
            if pos in self._valid_positions and pos not in blocked
        ]
        if not goals:
            return None

        path = self._a_star(start, set(goals), blocked)
        if path is None or len(path) < 2:
            return None
        return self._avoid_teammate_front_step(state, agent_index, path[1])

    def _next_step_to_floor(self, state, agent_index: int, target: tuple[int, int]) -> tuple[int, int] | None:
        player = state.players[agent_index]
        blocked = self._blocked_positions(state, agent_index)
        path = self._a_star(player.position, {target}, blocked)
        if path is None or len(path) < 2:
            return None
        return self._avoid_teammate_front_step(state, agent_index, path[1])

    def _avoid_teammate_front_step(
        self,
        state,
        agent_index: int,
        preferred: tuple[int, int],
    ) -> tuple[int, int]:
        player = state.players[agent_index]
        teammate = state.players[1 - agent_index]
        teammate_front = Action.move_in_direction(teammate.position, teammate.orientation)
        if preferred != teammate_front:
            return preferred

        blocked = self._blocked_positions(state, agent_index)
        options = []
        for direction in Direction.ALL_DIRECTIONS:
            nxt = Action.move_in_direction(player.position, direction)
            if nxt in self._valid_positions and nxt not in blocked and nxt != teammate_front:
                options.append(nxt)
        if not options:
            return preferred
        return min(options, key=lambda pos: self._manhattan(pos, preferred))

    def _a_star(
        self,
        start: tuple[int, int],
        goals: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        serial = count()
        frontier = []
        heappush(frontier, (0, next(serial), start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, _, current = heappop(frontier)
            if current in goals:
                return self._reconstruct_path(came_from, current)

            for direction in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(current, direction)
                if nxt not in self._valid_positions:
                    continue
                if nxt in blocked and nxt not in goals:
                    continue
                new_cost = cost_so_far[current] + 1
                if nxt in cost_so_far and new_cost >= cost_so_far[nxt]:
                    continue
                cost_so_far[nxt] = new_cost
                priority = new_cost + min(self._manhattan(nxt, goal) for goal in goals)
                heappush(frontier, (priority, next(serial), nxt))
                came_from[nxt] = current

        return None

    def _unstuck_action(self, state, mdp, agent_index: int):
        player = state.players[agent_index]
        blocked = self._blocked_positions(state, agent_index)
        options = []
        for direction in Direction.ALL_DIRECTIONS:
            nxt = Action.move_in_direction(player.position, direction)
            if nxt in self._valid_positions and nxt not in blocked:
                options.append(direction)
        if not options:
            return Action.STAY
        return options[int(self.rng.integers(0, len(options)))]

    def _blocked_positions(self, state, agent_index: int) -> set[tuple[int, int]]:
        if not self.avoid_teammate:
            return set()
        return {player.position for idx, player in enumerate(state.players) if idx != agent_index}

    # ------------------------------------------------------------------
    # Distances and geometry
    # ------------------------------------------------------------------

    def _refresh_layout_cache(self, mdp):
        layout_name = getattr(mdp, "layout_name", None)
        if layout_name == self._layout_name and self._valid_positions:
            return
        self._layout_name = layout_name
        self._valid_positions = set(mdp.get_valid_player_positions())
        self._distance_cache = {}

    def _best_feature_target(
        self,
        origin: tuple[int, int],
        targets: Iterable[tuple[int, int]],
    ) -> tuple[int, int] | None:
        targets = list(targets)
        if not targets:
            return None
        return min(targets, key=lambda target: self._feature_distance(origin, target))

    def _best_floor_target(
        self,
        origin: tuple[int, int],
        targets: Iterable[tuple[int, int]],
    ) -> tuple[int, int] | None:
        targets = list(targets)
        if not targets:
            return None
        return min(targets, key=lambda target: self._distance_between(origin, target))

    def _feature_distance(self, origin: tuple[int, int], feature: tuple[int, int]) -> int:
        goals = [pos for pos in self._adjacent_positions(feature) if pos in self._valid_positions]
        if not goals:
            return 999
        return min(self._distance_between(origin, goal) for goal in goals)

    def _distance_between(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]
        path = self._a_star(start, {goal}, blocked=set())
        dist = 999 if path is None else len(path) - 1
        self._distance_cache[key] = dist
        return dist

    @staticmethod
    def _reconstruct_path(came_from, current: tuple[int, int]) -> list[tuple[int, int]]:
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    @staticmethod
    def _adjacent_positions(pos: tuple[int, int]) -> list[tuple[int, int]]:
        return [Action.move_in_direction(pos, direction) for direction in Direction.ALL_DIRECTIONS]

    @staticmethod
    def _direction_from_to(a: tuple[int, int], b: tuple[int, int]):
        direction = (b[0] - a[0], b[1] - a[1])
        if direction not in Direction.ALL_DIRECTIONS:
            raise ValueError(f"Positions are not adjacent: {a} -> {b}")
        return direction
