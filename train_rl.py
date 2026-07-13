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

# ── Exact competition scenario definitions ─────────────────────────────────
# ── Competition scenarios (HIGH priority — 3 envs each) ───────────────────
# (layout, agent_ingredient, partner_ingredient, noise)
COMPETITION = [
    ("asymmetric_advantages",   "onion",   "onion",   0.00),   # Escenario 1
    ("coordination_ring",       "onion",   "onion",   0.25),   # Escenario 2
    ("counter_circuit",         "tomato",  "onion",   0.35),   # Escenario 3 — agent=tomato, partner=onion
]

# ── Extra layouts for generalization (LOW priority — 1 env each) ───────────
EXTRA = [
    ("cramped_room",            "onion",   "onion",   0.00),
    ("forced_coordination",     "onion",   "onion",   0.00),
    ("large_room",              "onion",   "onion",   0.00),
    ("small_corridor",          "onion",   "onion",   0.00),
    ("soup_coordination",       "onion",   "onion",   0.00),
    ("corridor",                "onion",   "onion",   0.00),
]

# Final list: competition x3 + extras x1  →  15 total envs
SCENARIOS = (COMPETITION * 3) + EXTRA
# ──────────────────────────────────────────────────────────────────────────

class CompetitionEnv(gym.Env):
    """Gym env that mirrors one of the 3 exact competition scenarios."""

    def __init__(self, env_id: int):
        super().__init__()
        layout, agent_ingredient, partner_ingredient, noise = SCENARIOS[env_id % len(SCENARIOS)]
        self.agent_ingredient = agent_ingredient

        self.mdp = OvercookedGridworld.from_layout_name(
            layout, old_dynamics=True  # must match evaluation YAMLs
        )
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(96,), dtype=np.float32
        )

        base_partner = GreedyFullTaskPolicy(ingredient=partner_ingredient)
        base_partner.set_mdp(self.mdp)

        if noise > 0:
            self.partner = EpsilonActionWrapper(
                base_partner,
                random_action_prob=noise,
            )
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
        print(f"  {layout} …")
        mdp = OvercookedGridworld.from_layout_name(layout)
        MediumLevelActionManager.from_pickle_or_compute(
            mdp, NO_COUNTERS_PARAMS, force_compute=False
        )
    print("Done!\n")

    # ── 15 parallel envs cycling through the 3 competition scenarios ───────
    # env_id 0,3,6,9,12  → Escenario 1 (asymmetric_advantages)
    # env_id 1,4,7,10,13 → Escenario 2 (coordination_ring)
    # env_id 2,5,8,11,14 → Escenario 3 (counter_circuit)
    N_ENVS = 15
    env = make_vec_env(
        lambda env_id=0: CompetitionEnv(env_id),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        # ── Tuned hyperparameters ──────────────────────────────────────────
        learning_rate=3e-4,
        n_steps=1024,          # steps per env before update  (15*1024=15 360 total)
        batch_size=512,        # larger batches → more stable gradients
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # encourage exploration early on
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),  # bigger network than default [64, 64]
    )

    total_steps = 8_000_000
    print(f"Training ONE general PPO agent on the 3 competition scenarios ({total_steps:,} steps)…")
    model.learn(total_timesteps=total_steps)
    model.save("models/ppo_general_agent")
    print("\nSaved → models/ppo_general_agent.zip")


if __name__ == "__main__":
    train()
