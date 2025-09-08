import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from g1_env import G1Env
from g1_cfg import g1_env_cfgs, g1_agent_cfg


def main():
    gs.init()

    env_cfg, obs_cfg, reward_cfg, command_cfg = g1_env_cfgs()
    env = G1Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    agent_cfg = g1_agent_cfg()
    runner = OnPolicyRunner(env, agent_cfg, device=agent_cfg["device"])
    runner.load(
        "checkpoints/g1_homie/2025-08-31_17-01-59/model_71999.pt",
        load_optimizer=False,
    )
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
