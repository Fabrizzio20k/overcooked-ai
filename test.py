import numpy as np
np.Inf = np.inf
np.NaN = np.nan

from overcooked_ai_py.mdp.actions import Action
def custom_action_sample(action_probs):
    idx = np.random.choice(len(Action.ALL_ACTIONS), p=action_probs)
    return Action.ALL_ACTIONS[idx]
Action.sample = staticmethod(custom_action_sample)

from stable_baselines3 import PPO
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import GreedyHumanModel, RandomAgent
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
def evaluate_model():
    model_path = "models/ppo_fcp_overcooked"
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print("Error al cargar el modelo (quizas aun no lo has entrenado):", e)
        return

    horizon = 400
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

    mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)
    partner = GreedyHumanModel(mlam)
    agent_idx = 0
    partner.set_agent_index(1 - agent_idx)

    env.reset()

    soups_delivered = 0
    first_soup_timestep = 0
    last_soup_timestep = 0
    timeouts = 0

    done = False
    timestep = 0

    while not done and timestep < horizon:
        obs = env.featurize_state_mdp(env.state)
        agent_obs = np.array(obs[agent_idx], dtype=np.float32)

        action, _ = model.predict(agent_obs, deterministic=True)
        partner_action, _ = partner.action(env.state)

        joint_action = [None, None]
        joint_action[agent_idx] = Action.ALL_ACTIONS[int(action)]
        joint_action[1 - agent_idx] = partner_action

        next_state, reward, done, info = env.step(joint_action)

        if reward > 0:
            soups_delivered += 1
            last_soup_timestep = timestep
            if soups_delivered == 1:
                first_soup_timestep = timestep

        timestep += 1

    score = 0
    if soups_delivered > 0:
        score = (
            (10000 * soups_delivered)
            + (10 * (horizon - last_soup_timestep))
            + (horizon - first_soup_timestep)
        )

    penalty = min(100 * timeouts, 5000)
    final_score = score - penalty

    print(f"--- RESULTADOS DE LA EVALUACION ---")
    print(f"Sopas entregadas: {soups_delivered}")
    print(f"Timestep 1ra Sopa: {first_soup_timestep}")
    print(f"Timestep Ultima Sopa: {last_soup_timestep}")
    print(f"Penalizaciones (Timeouts): {timeouts}")
    print(f"PUNTAJE FINAL (Formato Competencia): {final_score}")


if __name__ == "__main__":
    evaluate_model()
