import numpy as np
np.Inf = np.inf
np.NaN = np.nan

import os
import argparse
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

class EscenarioTrainEnv(gym.Env):
    """Custom gym environment to train a PPO agent for any of the 3 scenarios."""
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        
        if scenario == 1:
            self.mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
            base_partner = GreedyFullTaskPolicy(ingredient="onion")
            base_partner.set_mdp(self.mdp)
            self.partner = base_partner  # No noise
        elif scenario == 2:
            self.mdp = OvercookedGridworld.from_layout_name("coordination_ring")
            base_partner = GreedyFullTaskPolicy(ingredient="onion")
            base_partner.set_mdp(self.mdp)
            self.partner = EpsilonActionWrapper(base_partner, random_action_prob=0.25)
        elif scenario == 3:
            self.mdp = OvercookedGridworld.from_layout_name("counter_circuit")
            base_partner = GreedyFullTaskPolicy(ingredient="onion")
            base_partner.set_mdp(self.mdp)
            self.partner = EpsilonActionWrapper(base_partner, random_action_prob=0.35)
        else:
            raise ValueError(f"Invalid scenario: {scenario}")
            
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(96,), dtype=np.float32)
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

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for specific scenarios.")
    parser.add_argument(
        "--escenario", 
        type=int, 
        required=True,
        choices=[1, 2, 3],
        help="The scenario number to train (1, 2, or 3)."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1000000, 
        help="Total steps to train the agent."
    )
    args = parser.parse_args()

    scenario = args.escenario
    steps = args.steps

    print(f"\n[Training] Setting up training for Escenario {scenario} for {steps:,} steps...")
    
    # Ensure models/ dir exists
    os.makedirs("models", exist_ok=True)

    env = make_vec_env(lambda: EscenarioTrainEnv(scenario), n_envs=16, vec_env_cls=SubprocVecEnv)
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
    
    print(f"[Training] Starting learning loop...")
    model.learn(total_timesteps=steps)
    
    save_path = f"models/ppo_escenario{scenario}"
    model.save(save_path)
    print(f"[Training] Completed! Saved PPO model to: '{save_path}.zip'\n")

if __name__ == "__main__":
    main()
