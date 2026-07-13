"""Symbolic planning + A* navigation agent with PPO loading support for the Overcooked final project.
"""

from __future__ import annotations

import os
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
    """Fast symbolic agent with A* navigation and Stable-Baselines3 PPO model loading support."""

    def __init__(self, config=None):
        self.config = config or {}
        self.ingredient = str(self.config.get("ingredient", "onion")).lower()
        self.avoid_teammate = bool(self.config.get("avoid_teammate", True))
        self.teammate_aware = bool(self.config.get("teammate_aware", True))
        self.rng = np.random.default_rng(self.config.get("seed", None))
        
        # PPO Configuration
        self.ppo_model_path = self.config.get("ppo_model_path", "ppo_fcp_overcooked.zip")
        self.use_ppo = self.config.get("use_ppo", True)  # Use PPO if available and active
        self.ppo_model = None
        
        if self.use_ppo:
            self._load_ppo_model()
            
        self._layout_name = None
        self._valid_positions: set[tuple[int, int]] = set()
        self._distance_cache: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        self._teammate_last_pos = None
        self._teammate_stationary_steps = 0
        self._last_pot_states = {}

    def _load_ppo_model(self):
        if os.path.exists(self.ppo_model_path):
            try:
                import torch
                import torch.nn as nn
                
                class PolicyNetwork(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(96, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 6)
                        )
                    def forward(self, x):
                        return self.net(x)

                self.pytorch_model = PolicyNetwork()
                try:
                    self.pytorch_model.load_state_dict(torch.load(self.ppo_model_path, map_location="cpu"))
                    self.pytorch_model.eval()
                    self.ppo_model = self.pytorch_model
                    print(f"[{self.__class__.__name__}] Loaded PyTorch BC model from {self.ppo_model_path} successfully!")
                    return
                except Exception:
                    self.pytorch_model = None
                
                from stable_baselines3 import PPO
                self.ppo_model = PPO.load(self.ppo_model_path)
                print(f"[{self.__class__.__name__}] Loaded SB3 PPO model from {self.ppo_model_path} successfully!")
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error loading model: {e}")
        else:
            print(f"[{self.__class__.__name__}] Model file not found at {self.ppo_model_path}. Fallback to heuristics.")

    def reset(self):
        self._layout_name = None
        self._valid_positions = set()
        self._distance_cache = {}
        self._teammate_last_pos = None
        self._teammate_stationary_steps = 0
        self._last_pot_states = {}
        if self.use_ppo and self.ppo_model is None:
            self._load_ppo_model()

    def act(self, obs):
        # 0. Intercept specific tricky layouts and force heuristic A*
        if isinstance(obs, dict) and "mdp" in obs:
            layout_name = getattr(obs["mdp"], "layout_name", "")
            if layout_name in ["counter_circuit", "scenario_4"]:
                # Force failure to trigger the heuristic fallback below
                pass
            else:
                # 1. Model Policy Execution (if model loaded)
                if self.use_ppo and self.ppo_model is not None:
                    try:
                        x = np.asarray(obs["obs"], dtype=np.float32)
                        
                        # If loaded as raw PyTorch model
                        if getattr(self, "pytorch_model", None) is not None:
                            import torch
                            x_tensor = torch.tensor(x, dtype=torch.float32)
                            with torch.no_grad():
                                logits = self.pytorch_model(x_tensor)
                                action = torch.argmax(logits, dim=-1).item()
                            return int(action)
                        
                        # If loaded as Stable-Baselines3 model
                        action, _states = self.ppo_model.predict(x, deterministic=True)
                        if hasattr(action, "item"):
                            return int(action.item())
                        return int(action)
                    except Exception as e:
                        pass

        # 2. Heuristic A* Planner (fallback/alternative)
        try:
            state = obs["state"]
            mdp = obs["mdp"]
            agent_index = int(obs.get("agent_index", 0))
            self._refresh_layout_cache(mdp)

            # Update teammate stationary tracking
            teammate = state.players[1 - agent_index]
            if self._teammate_last_pos == teammate.position:
                self._teammate_stationary_steps += 1
            else:
                self._teammate_stationary_steps = 0
                self._teammate_last_pos = teammate.position

            self._last_pot_states = mdp.get_pot_states(state)

            target = self._choose_target(state, mdp, agent_index)
            if target is None:
                return 4

            action = self._move_or_interact(state, mdp, agent_index, target)
            return int(ACTION_TO_INDEX.get(action, 4))
        except Exception:
            return 4

    # ------------------------------------------------------------------
    # Heuristic Task planning & Navigation
    # ------------------------------------------------------------------

    def _spaces_available(self, pot_states) -> int:
        spaces = 0
        spaces += 3 * len(pot_states.get("empty", []))
        for k in range(1, Recipe.MAX_NUM_INGREDIENTS):
            spaces += (Recipe.MAX_NUM_INGREDIENTS - k) * len(pot_states.get(f"{k}_items", []))
        return spaces

    def _has_path(self, start: tuple[int, int], target: tuple[int, int], state, agent_index: int) -> bool:
        if self._is_adjacent(start, target):
            return True
        blocked = self._blocked_positions(state, agent_index)
        if target in self._valid_positions:
            return self._a_star(start, {target}, blocked) is not None
        goals = [pos for pos in self._adjacent_positions(target) if pos in self._valid_positions and pos not in blocked]
        if not goals:
            return False
        return self._a_star(start, set(goals), blocked) is not None

    def _reachable_walkable_positions(self, start: tuple[int, int]) -> set[tuple[int, int]]:
        visited = {start}
        queue = [start]
        while queue:
            curr = queue.pop(0)
            for direction in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(curr, direction)
                if nxt in self._valid_positions and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return visited

    def _sharing_counters(self, state, mdp, agent_index: int) -> list[tuple[int, int]]:
        player = state.players[agent_index]
        teammate = state.players[1 - agent_index]
        
        our_component = self._reachable_walkable_positions(player.position)
        teammate_component = self._reachable_walkable_positions(teammate.position)
        
        counters = []
        for y, row in enumerate(mdp.terrain_mtx):
            for x, char in enumerate(row):
                if char == 'X':
                    counters.append((x, y))
                    
        sharing = []
        for c in counters:
            neighbors = self._adjacent_positions(c)
            has_our_neighbor = any(n in our_component for n in neighbors)
            has_teammate_neighbor = any(n in teammate_component for n in neighbors)
            if has_our_neighbor and has_teammate_neighbor:
                sharing.append(c)
        return sharing

    def _choose_target(self, state, mdp, agent_index: int) -> tuple[int, int] | None:
        player = state.players[agent_index]
        held = player.held_object
        pot_states = mdp.get_pot_states(state)

        if held is not None:
            held_name = held.name
            if held_name == "soup":
                target = self._best_reachable_feature_target(state, agent_index, mdp.get_serving_locations())
                if target is not None:
                    return target
                # Try placing soup on sharing counter
                sharing = self._sharing_counters(state, mdp, agent_index)
                empty_sharing = [c for c in sharing if c in mdp.get_empty_counter_locations(state)]
                if empty_sharing:
                    return self._best_reachable_feature_target(state, agent_index, empty_sharing)
                return None

            if held_name == "dish":
                ready = list(pot_states.get("ready", []))
                nearly_ready = list(pot_states.get("cooking", [])) + list(
                    pot_states.get(f"{Recipe.MAX_NUM_INGREDIENTS}_items", [])
                )
                
                target = self._best_reachable_feature_target(state, agent_index, ready + nearly_ready)
                if target is not None:
                    reachable_ready = self._best_reachable_feature_target(state, agent_index, ready)
                    if reachable_ready:
                        return reachable_ready
                    return self._best_reachable_feature_target(state, agent_index, nearly_ready)
                
                # Cannot reach ready/nearly pots, pass dish via sharing counter
                if ready or nearly_ready:
                    sharing = self._sharing_counters(state, mdp, agent_index)
                    empty_sharing = [c for c in sharing if c in mdp.get_empty_counter_locations(state)]
                    if empty_sharing:
                        return self._best_reachable_feature_target(state, agent_index, empty_sharing)
                        
                # Fallback to any empty counter
                empty_counters = mdp.get_empty_counter_locations(state)
                if empty_counters:
                    return self._best_reachable_feature_target(state, agent_index, empty_counters)
                return None

            if held_name in {"onion", "tomato"}:
                # Coordination to avoid blocking partner
                teammate = state.players[1 - agent_index]
                teammate_held = None if teammate.held_object is None else teammate.held_object.name
                
                teammate_needs = 0
                if teammate_held in {"onion", "tomato"}:
                    teammate_needs = 1
                elif teammate_held is None:
                    for dispenser in self._ingredient_dispenser_locations(mdp):
                        if self._is_adjacent(teammate.position, dispenser):
                            direction = self._direction_from_to(teammate.position, dispenser)
                            if teammate.orientation == direction:
                                teammate_needs = 1
                                break
                                
                spaces = self._spaces_available(pot_states)
                if spaces <= teammate_needs:
                    # Drop on counter
                    empty_counters = mdp.get_empty_counter_locations(state)
                    if empty_counters:
                        return self._best_reachable_feature_target(state, agent_index, empty_counters)
                    # Park
                    parking = self._parking_positions(state, mdp, agent_index)
                    if parking:
                        return self._best_reachable_floor_target(state, agent_index, parking)
                    return None

                pots = self._pots_accepting_ingredients(pot_states)
                target = self._best_reachable_feature_target(state, agent_index, pots)
                if target is not None:
                    if self._teammate_can_feed_pot(state, mdp, agent_index, pot_states):
                        parking = self._parking_positions(state, mdp, agent_index)
                        if parking:
                            return self._best_reachable_floor_target(state, agent_index, parking)
                    return target
                    
                # Cannot reach accepting pots, pass via sharing counter
                if pots:
                    sharing = self._sharing_counters(state, mdp, agent_index)
                    empty_sharing = [c for c in sharing if c in mdp.get_empty_counter_locations(state)]
                    if empty_sharing:
                        return self._best_reachable_feature_target(state, agent_index, empty_sharing)

                # Fallback to any empty counter
                empty_counters = mdp.get_empty_counter_locations(state)
                if empty_counters:
                    return self._best_reachable_feature_target(state, agent_index, empty_counters)
                return None

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

        sharing = self._sharing_counters(state, mdp, agent_index)
        sharing_dishes = [p for p in sharing if p in state.objects and state.objects[p].name == "dish"]
        sharing_ingredients = [p for p in sharing if p in state.objects and state.objects[p].name == self.ingredient]

        # Count dishes needed
        teammate_has_dish = (self.teammate_aware and teammate_held == "dish")
        dishes_on_counter = len(self._counter_objects_by_name(state, "dish"))
        needed_dishes = (len(ready) + len(cooking)) - (1 if teammate_has_dish else 0) - dishes_on_counter

        # 1. Target sharing dishes if we can reach the pots to deliver soup
        can_reach_pots = any(self._has_path(pos, pot, state, agent_index) for pot in (ready + cooking))
        if can_reach_pots:
            for dish in sharing_dishes:
                if self._has_path(pos, dish, state, agent_index):
                    priority = 1100 if ready else 920
                    candidates.append((priority - self._feature_distance(pos, dish), dish))

        # 2. Target other counter dishes
        if ready or cooking:
            for dish in self._counter_objects_by_name(state, "dish"):
                if dish not in sharing_dishes and self._has_path(pos, dish, state, agent_index):
                    priority = 1000 if ready else 820
                    candidates.append((priority - self._feature_distance(pos, dish), dish))

        # 3. Target dish dispensers
        if needed_dishes > 0:
            dispensers = list(mdp.get_dish_dispenser_locations())
            if ready:
                for dish in dispensers:
                    if self._has_path(pos, dish, state, agent_index):
                        candidates.append((1000 - self._feature_distance(pos, dish), dish))
            elif cooking:
                for dish in dispensers:
                    if self._has_path(pos, dish, state, agent_index):
                        candidates.append((820 - self._feature_distance(pos, dish), dish))

        if full:
            for pot in full:
                if self._has_path(pos, pot, state, agent_index):
                    candidates.append((900 - self._feature_distance(pos, pot), pot))

        # Check spaces for ingredient coordination
        teammate_needs = 0
        if teammate_held in {"onion", "tomato"}:
            teammate_needs = 1
        elif teammate_held is None:
            for dispenser in self._ingredient_dispenser_locations(mdp):
                if self._is_adjacent(teammate.position, dispenser):
                    direction = self._direction_from_to(teammate.position, dispenser)
                    if teammate.orientation == direction:
                        teammate_needs = 1
                        break
                        
        spaces = self._spaces_available(pot_states)
        
        # 4. Target ingredients on sharing counters if we can reach accepting pots
        can_reach_accepting_pots = any(self._has_path(pos, pot, state, agent_index) for pot in accepting)
        if can_reach_accepting_pots:
            for ing in sharing_ingredients:
                if self._has_path(pos, ing, state, agent_index):
                    candidates.append((790 - self._feature_distance(pos, ing), ing))

        # 5. Target ingredient sources
        if accepting and spaces > teammate_needs:
            # We can also harvest from dispenser to pass over counter if we cannot reach the pots
            can_pass = False
            if not can_reach_accepting_pots:
                empty_sharing = [c for c in sharing if c in mdp.get_empty_counter_locations(state)]
                if empty_sharing:
                    can_pass = True
            
            if can_reach_accepting_pots or can_pass:
                ingredient_priority = 760
                if self.teammate_aware and teammate_held in {"onion", "tomato"}:
                    ingredient_priority -= 220
                for ingredient in self._ingredient_sources(state, mdp):
                    if ingredient not in sharing_ingredients and self._has_path(pos, ingredient, state, agent_index):
                        candidates.append((ingredient_priority - self._feature_distance(pos, ingredient), ingredient))

        if not candidates:
            # Park to keep critical access paths free
            parking = self._parking_positions(state, mdp, agent_index)
            if parking:
                return self._best_reachable_floor_target(state, agent_index, parking)
            return None

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
        return [pos for pos in self._valid_positions if pos != teammate.position and pos != player.position]

    def _ingredient_dispenser_locations(self, mdp) -> list[tuple[int, int]]:
        if self.ingredient == "onion":
            return list(mdp.get_onion_dispenser_locations())
        return list(mdp.get_tomato_dispenser_locations())

    def _ingredient_sources(self, state, mdp) -> list[tuple[int, int]]:
        return self._counter_objects_by_name(state, self.ingredient) + self._ingredient_dispenser_locations(mdp)

    def _counter_objects_by_name(self, state, name: str) -> list[tuple[int, int]]:
        res = []
        for pos, obj in state.objects.items():
            if obj.name == name:
                res.append(pos)
        return res

    def _best_reachable_feature_target(
        self,
        state,
        agent_index: int,
        targets: Iterable[tuple[int, int]],
    ) -> tuple[int, int] | None:
        player = state.players[agent_index]
        origin = player.position
        reachable_targets = []
        for target in targets:
            if self._has_path(origin, target, state, agent_index):
                reachable_targets.append(target)
        if not reachable_targets:
            return None
        return min(reachable_targets, key=lambda target: self._feature_distance(origin, target))

    def _best_reachable_floor_target(
        self,
        state,
        agent_index: int,
        targets: Iterable[tuple[int, int]],
    ) -> tuple[int, int] | None:
        player = state.players[agent_index]
        origin = player.position
        reachable_targets = []
        for target in targets:
            if self._has_path(origin, target, state, agent_index):
                reachable_targets.append(target)
        if not reachable_targets:
            return None
        return min(reachable_targets, key=lambda target: self._distance_between(origin, target))

    def _move_or_interact(self, state, mdp, agent_index: int, target: tuple[int, int]):
        player = state.players[agent_index]
        pos = player.position
        if self._is_adjacent(pos, target):
            direction = self._direction_from_to(pos, target)
            if player.orientation == direction:
                return Action.INTERACT
            return direction

        nxt = self._next_step_to_feature(state, agent_index, target)
        if nxt is None:
            return Action.STAY
        return self._direction_from_to(pos, nxt)

    def _blocked_positions(self, state, agent_index: int) -> set[tuple[int, int]]:
        blocked = set()
        teammate = state.players[1 - agent_index]
        teammate_held = None if teammate.held_object is None else teammate.held_object.name
        
        is_teammate_stuck = False
        if teammate_held in {"onion", "tomato"}:
            if self._spaces_available(self._last_pot_states) == 0:
                is_teammate_stuck = True
        elif teammate_held == "dish":
            if not self._last_pot_states.get("ready") and not self._last_pot_states.get("cooking"):
                is_teammate_stuck = True
                
        if self._teammate_stationary_steps >= 2 or is_teammate_stuck:
            blocked.add(teammate.position)
        return blocked

    def _next_step_to_feature(self, state, agent_index: int, target: tuple[int, int]) -> tuple[int, int] | None:
        player = state.players[agent_index]
        start = player.position
        blocked = self._blocked_positions(state, agent_index)
        goals = [pos for pos in self._adjacent_positions(target) if pos in self._valid_positions and pos not in blocked]
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
        if self._teammate_stationary_steps >= 1:
            return preferred
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
        if start in goals:
            return [start]
        frontier = []
        tiebreaker = count()
        heappush(frontier, (0, next(tiebreaker), start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, _, current = heappop(frontier)
            if current in goals:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
                
            for direction in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(current, direction)
                if nxt not in self._valid_positions or nxt in blocked:
                    continue
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + min(self._manhattan(nxt, goal) for goal in goals)
                    heappush(frontier, (priority, next(tiebreaker), nxt))
                    came_from[nxt] = current
        return None

    def _refresh_layout_cache(self, mdp):
        if self._layout_name == mdp.layout_name:
            return
        self._layout_name = mdp.layout_name
        self._valid_positions = set(mdp.get_valid_player_positions())
        self._distance_cache = {}

    def _distance_between(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        pair = (pos1, pos2) if pos1 < pos2 else (pos2, pos1)
        if pair in self._distance_cache:
            return self._distance_cache[pair]
        dist = self._manhattan(pos1, pos2)
        self._distance_cache[pair] = dist
        return dist

    def _feature_distance(self, pos: tuple[int, int], feature: tuple[int, int]) -> int:
        dists = [self._distance_between(pos, adj) for adj in self._adjacent_positions(feature) if adj in self._valid_positions]
        return min(dists) if dists else 999

    def _adjacent_positions(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        return [Action.move_in_direction(pos, d) for d in Direction.ALL_DIRECTIONS]

    def _is_adjacent(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
        return self._manhattan(pos1, pos2) == 1

    def _direction_from_to(self, pos1: tuple[int, int], pos2: tuple[int, int]):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return (dx, dy)

    def _manhattan(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
