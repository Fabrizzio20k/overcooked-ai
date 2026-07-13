import numpy as np
np.Inf = np.inf
np.NaN = np.nan

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

from policies.basic_policies import GreedyFullTaskPolicy
from src.policy_wrappers import EpsilonActionWrapper

class Escenario2TrainEnv(gym.Env):
    """Custom gym environment to train a PPO agent for Escenario 2."""
    def __init__(self):
        super().__init__()
        self.mdp = OvercookedGridworld.from_layout_name("coordination_ring")
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(96,), dtype=np.float32)
        
        # Teammate: greedy_full_task with 25% random actions (sticky/noisy actions)
        base_partner = GreedyFullTaskPolicy(ingredient="onion")
        base_partner.set_mdp(self.mdp)
        self.partner = EpsilonActionWrapper(base_partner, random_action_prob=0.25)
        self.agent_idx = 0

    def seed(self, seed=None):
        pass

    def reset(self):
        self.base_env.reset()
        self.agent_idx = np.random.choice([0, 1])
        self.partner.set_agent_index(1 - self.agent_idx)
        obs = self.base_env.featurize_state_mdp(self.base_env.state)
        return np.array(obs[self.agent_idx], dtype=np.float32)

    def step(self, action):
        partner_action, _ = self.partner.action(self.base_env.state)
        joint_action = [None, None]
        joint_action[self.agent_idx] = Action.ALL_ACTIONS[int(action)]
        joint_action[1 - self.agent_idx] = partner_action
        
        next_state, reward, done, info = self.base_env.step(joint_action)
        obs = self.base_env.featurize_state_mdp(self.base_env.state)
        return np.array(obs[self.agent_idx], dtype=np.float32), reward, done, info

def train():
    env = make_vec_env(lambda: Escenario2TrainEnv(), n_envs=16)
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
    print("Training PPO model for Escenario 2 for 1,000,000 steps...")
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_escenario2")
    print("Saved model to ppo_escenario2.zip!")

if __name__ == "__main__":
    train()
