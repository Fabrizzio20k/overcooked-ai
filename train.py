import numpy as np
np.Inf = np.inf
np.NaN = np.nan

from overcooked_ai_py.mdp.actions import Action
def custom_action_sample(action_probs):
    idx = np.random.choice(len(Action.ALL_ACTIONS), p=action_probs)
    return Action.ALL_ACTIONS[idx]
Action.sample = staticmethod(custom_action_sample)

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
class FCPOvercookedEnv(gym.Env):
    def __init__(self, layout_name="cramped_room"):
        super(FCPOvercookedEnv, self).__init__()
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(96,), dtype=np.float32)

        self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)
        self.partners = [RandomAgent(), GreedyHumanModel(self.mlam)]
        self.current_partner = None
        self.agent_idx = 0

    def seed(self, seed=None):
        pass

    def reset(self):
        self.base_env.reset()
        self.current_partner = np.random.choice(self.partners)
        self.agent_idx = np.random.choice([0, 1])
        self.current_partner.set_agent_index(1 - self.agent_idx)

        obs = self.base_env.featurize_state_mdp(self.base_env.state)
        return np.array(obs[self.agent_idx], dtype=np.float32)

    def step(self, action):
        partner_action, _ = self.current_partner.action(self.base_env.state)
        if isinstance(partner_action, tuple):
            partner_action = partner_action[0]

        joint_action = [None, None]
        joint_action[self.agent_idx] = action
        joint_action[1 - self.agent_idx] = partner_action

        next_state, reward, done, info = self.base_env.step(joint_action)
        obs = self.base_env.featurize_state_mdp(self.base_env.state)

        return np.array(obs[self.agent_idx], dtype=np.float32), reward, done, info


def train_fcp():
    env = make_vec_env(lambda: FCPOvercookedEnv(layout_name="asymmetric_advantages"), n_envs=4)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )
    model.learn(total_timesteps=1_000_000)
    model.save("models/ppo_fcp_overcooked")


if __name__ == "__main__":
    train_fcp()
