import numpy as np
np.Inf = np.inf
np.NaN = np.nan

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

from policies.basic_policies import GreedyFullTaskPolicy
from src.policy_wrappers import EpsilonActionWrapper

# ── Competition scenarios (HIGH priority — 3 envs each = 12 envs) ─────────
# (layout_or_file, agent_ingredient, partner_ingredient, noise)
# Use "file:path/to/file.layout" for custom layouts
COMPETITION = [
    ("asymmetric_advantages",              "onion",  "onion",  0.00),   # Escenario 1
    ("coordination_ring",                  "onion",  "onion",  0.25),   # Escenario 2
    ("counter_circuit",                    "tomato", "onion",  0.35),   # Escenario 3
    ("file:configs/layouts/scenario_4.layout", "onion", "onion", 1.00), # Escenario 4 — random_motion partner
]

# ── Generalization layouts (LOW priority — 1 env each = 12 envs) ──────────
EXTRA = [
    ("cramped_room",              "onion",  "onion",  0.00),
    ("forced_coordination",       "onion",  "onion",  0.00),
    ("large_room",                "onion",  "onion",  0.00),
    ("small_corridor",            "onion",  "onion",  0.20),
    ("soup_coordination",         "onion",  "onion",  0.00),
    ("corridor",                  "onion",  "onion",  0.20),
    ("five_by_five",              "onion",  "onion",  0.00),
    ("cramped_room_tomato",       "tomato", "tomato", 0.00),
    ("asymmetric_advantages_tomato", "tomato", "tomato", 0.00),
    ("forced_coordination_tomato","tomato", "tomato", 0.00),
    ("bottleneck",                "onion",  "onion",  0.20),
    ("unident",                   "onion",  "onion",  0.20),
]

# competition x3 + extras x1  →  24 total envs
SCENARIOS = (COMPETITION * 3) + EXTRA
# ──────────────────────────────────────────────────────────────────────────

class CompetitionEnv(gym.Env):
    """Gym env for competition scenarios — supports named and file-based layouts."""

    def __init__(self, env_id: int):
        super().__init__()
        layout, agent_ingredient, partner_ingredient, noise = SCENARIOS[env_id % len(SCENARIOS)]
        self.agent_ingredient = agent_ingredient

        # Support both named layouts and file-based custom layouts
        if layout.startswith("file:"):
            import json, re
            layout_path = layout[5:]  # strip "file:"
            with open(layout_path) as f:
                raw = f.read()
            layout_dict = json.loads(re.sub(r':\s*None', ': null', raw))
            grid = "\n".join(line.strip() for line in layout_dict["grid"].strip().splitlines())
            layout_dict.pop("grid")
            layout_dict["layout_name"] = "scenario_4"
            self.mdp = OvercookedGridworld.from_grid(
                layout_grid=grid,
                base_layout_params=layout_dict,
                params_to_overwrite={"old_dynamics": True},
            )
        else:
            self.mdp = OvercookedGridworld.from_layout_name(
                layout, old_dynamics=True
            )

        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(96,), dtype=np.float32
        )

        base_partner = GreedyFullTaskPolicy(ingredient=partner_ingredient)
        base_partner.set_mdp(self.mdp)
        if noise > 0:
            self.partner = EpsilonActionWrapper(base_partner, random_action_prob=noise)
        else:
            self.partner = base_partner

        self.agent_idx = 0

    def seed(self, seed=None):
        pass

    def reset(self):
        self.base_env.reset()
        # Randomise which side the agent spawns on so it learns both positions
        self.agent_idx = int(np.random.randint(0, 2))
        self.partner.set_agent_index(1 - self.agent_idx)
        obs = self.base_env.featurize_state_mdp(self.base_env.state)
        return np.array(obs[self.agent_idx], dtype=np.float32)

    def step(self, action):
        partner_action, _ = self.partner.action(self.base_env.state)
        joint_action = [None, None]
        joint_action[self.agent_idx] = Action.ALL_ACTIONS[int(action)]
        joint_action[1 - self.agent_idx] = partner_action

        _, reward, done, info = self.base_env.step(joint_action)
        obs = self.base_env.featurize_state_mdp(self.base_env.state)
        return np.array(obs[self.agent_idx], dtype=np.float32), reward, done, info


def train():
    os.makedirs("models", exist_ok=True)

    # ── Pre-compute motion planners to prevent race condition ──────────────
    print("Pre-computing motion planners sequentially...")
    from overcooked_ai_py.planning.planners import (
        MediumLevelActionManager, NO_COUNTERS_PARAMS,
    )
    for layout, _, _, _ in set(SCENARIOS):
        if layout.startswith("file:"):
            continue  # custom file layouts skip pre-computation
        print(f"  {layout} …")
        mdp = OvercookedGridworld.from_layout_name(layout)
        MediumLevelActionManager.from_pickle_or_compute(
            mdp, NO_COUNTERS_PARAMS, force_compute=False
        )
    print("Done!\n")

    # ── 24 parallel envs: 12 competition (4x each) + 12 generalization ────
    N_ENVS = 24
    env = make_vec_env(
        lambda env_id=0: CompetitionEnv(env_id),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,          # 24*1024 = 24 576 steps per update
        batch_size=512,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    total_steps = 10_000_000   # ~90 min on Khipu
    print(f"Training general PPO agent on {len(set(SCENARIOS))} unique layouts ({total_steps:,} steps)…")
    model.learn(total_timesteps=total_steps)
    model.save("models/ppo_general_agent")
    print("\nSaved → models/ppo_general_agent.zip")


if __name__ == "__main__":
    train()
