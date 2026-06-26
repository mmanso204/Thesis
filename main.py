from envs.environment_multi import HouseEnv
from agent import Agent
from helper_functions.goals import GOALS
from multigrid.core.world_object import Floor

ontology_path = "/Users/m.manso/Downloads/thesisont_updated-2.owl"

active_goal = GOALS["collect_trash"]  

env = HouseEnv(
    num_agents=1,
    goal=active_goal,
    render_mode="human",
    max_steps=500,
    width=32,
    height=23,
)

a1 = Agent(ontology_path)
a1.set_goal(active_goal)   
obs, info = env.reset()

while True:
    print("Action space:")
    for action in env.action_space.keys():
        print(action)
    action_input = input("Enter an action: ")

    obs, rewards, terminations, truncations, infos = env.step({0: int(action_input)})

    reward = rewards[0]
    terminated = terminations[0]
    truncated = truncations[0]
    info = infos[0]

    agent_obs = obs[0]
    agent_pos = env.agents[0].state.pos
    obs_data = a1.observations(agent_obs, agent_pos, env)
    a1.observations_to_ont(obs_data, env)

    for (x, y) in obs_data["Visible_world_coords"]:
        cell = env.grid.get(x, y)
        if cell is None or isinstance(cell, Floor):
            env.grid.set(x, y, Floor("blue"))

    if terminated or truncated:
        obs, info = env.reset()
        a1.reset()

    env.render()