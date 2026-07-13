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

# Fix numpy sampling issues
def custom_action_sample(action_probs):
    idx = np.random.choice(len(Action.ALL_ACTIONS), p=action_probs)
    return Action.ALL_ACTIONS[idx]
Action.sample = staticmethod(custom_action_sample)

class MultiMapTrainEnv(gym.Env):
    """Custom gym environment to train a PPO agent across multiple maps and partners."""
    def __init__(self, env_id):
        super().__init__()
        
        self.layouts = [
            "cramped_room",
            "asymmetric_advantages",
            "coordination_ring",
            "forced_coordination",
            "counter_circuit"
        ]
        self.env_id = env_id
        self.layout_name = self.layouts[env_id % len(self.layouts)]
        
        self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(96,), dtype=np.float32)
        
        # Configure partner
        base_partner = GreedyFullTaskPolicy(ingredient="onion")
        base_partner.set_mdp(self.mdp)
        
        # Dynamic partner noise based on layout
        noise = 0.0
        if self.layout_name == "coordination_ring":
            noise = 0.25
        elif self.layout_name == "counter_circuit":
            noise = 0.35
            
        if noise > 0:
            self.partner = EpsilonActionWrapper(base_partner, random_action_prob=noise)
        else:
            self.partner = base_partner
            
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
    os.makedirs("models", exist_ok=True)
    # 16 parallel environments. Each will load one of the 5 layouts cyclically.
    env = make_vec_env(lambda env_id=0: MultiMapTrainEnv(env_id), n_envs=16, vec_env_cls=SubprocVecEnv)
    
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
    print("Training a SINGLE general PPO agent across all 5 maps simultaneously...")
    model.learn(total_timesteps=1000000)
    model.save("models/ppo_general_agent")
    print("Saved general model to models/ppo_general_agent.zip!")

if __name__ == "__main__":
    train()
