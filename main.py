from policy.evader_policy import EvaderPolicy
import hydra
from omegaconf import DictConfig
from mpe2 import simple_tag_v3
from pettingzoo import ParallelEnv


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # environment initialization
    render_mode = "human" if cfg.env.render else None

    env: ParallelEnv = simple_tag_v3.parallel_env(
        num_good=cfg.env.num_good,
        num_adversaries=cfg.env.num_adversaries,
        num_obstacles=cfg.env.num_obstacles,
        max_cycles=cfg.env.max_cycles,
        continuous_actions=cfg.env.continuous_actions,
        dynamic_rescaling=cfg.env.dynamic_rescaling,
        curriculum=cfg.env.curriculum,
        terminate_on_success=cfg.env.terminate_on_success,
        render_mode=render_mode,
    )

    if cfg.env.render:
        env.metadata["render_fps"] = cfg.env.render_fps

    obs, _ = env.reset(seed=cfg.seed)

    # policy initialization
    evader_policy = EvaderPolicy(
        policy_mode="scripted",
        distance_weight=cfg.evader_policy.distance_weight,
        velocity_weight=cfg.evader_policy.velocity_weight,
        hunter_priority=cfg.evader_policy.hunter_priority,
        boundary_priority=cfg.evader_policy.boundary_priority,
        hunter_threat_radius=cfg.evader_policy.hunter_threat_radius,
        boundary_threshold=cfg.evader_policy.boundary_threshold,
        boundary_margin=cfg.evader_policy.boundary_margin,
    )

    while env.agents:
        actions = {
            agent: evader_policy.sample(obs['agent_0']) if agent == 'agent_0' else env.action_space(agent).sample()
            for agent in env.agents
        }

        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        print(obs['agent_0'])
        print(f'reward = {rewards["agent_0"]}')

    env.close()


if __name__ == "__main__":
    main()
