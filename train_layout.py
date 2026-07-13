import numpy as np
np.Inf = np.inf
np.NaN = np.nan

import os
import argparse
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent, GreedyHumanModel
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action

# Fix numpy sampling issues for overcooked library
def custom_action_sample(action_probs):
    idx = np.random.choice(len(Action.ALL_ACTIONS), p=action_probs)
    return Action.ALL_ACTIONS[idx]
Action.sample = staticmethod(custom_action_sample)

class GeneralTrainEnv(gym.Env):
    """Custom gym environment to train a PPO agent for a specific layout."""
    def __init__(self, layout_name):
        super().__init__()
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=400)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(96,), dtype=np.float32)
        
        # Build training partners
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
        joint_action = [None, None]
        joint_action[self.agent_idx] = Action.ALL_ACTIONS[int(action)]
        joint_action[1 - self.agent_idx] = partner_action
        
        next_state, reward, done, info = self.base_env.step(joint_action)
        obs = self.base_env.featurize_state_mdp(self.base_env.state)
        return np.array(obs[self.agent_idx], dtype=np.float32), reward, done, info

def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent on any Overcooked layout.")
    parser.add_argument(
        "--layout", 
        type=str, 
        default="cramped_room",
        choices=["cramped_room", "asymmetric_advantages", "coordination_ring", "forced_coordination", "counter_circuit"],
        help="Name of the layout to train on."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1000000, 
        help="Total steps to train the agent."
    )
    args = parser.parse_args()

    layout = args.layout
    steps = args.steps

    print(f"\n[Training] Setting up training for layout: '{layout}' for {steps:,} steps...")
    
    # Ensure models/ dir exists
    os.makedirs("models", exist_ok=True)

    env = make_vec_env(lambda: GeneralTrainEnv(layout), n_envs=16)
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
    
    print(f"[Training] Starting PPO learning loop...")
    model.learn(total_timesteps=steps)
    
    save_path = f"models/ppo_{layout}"
    model.save(save_path)
    print(f"[Training] Completed! Saved PPO model weights to: '{save_path}.zip'\n")

if __name__ == "__main__":
    main()
