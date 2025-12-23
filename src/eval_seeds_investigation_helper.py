import logging

from nsai_experiments.zoning_game.zg_policy import play_one_game
from nsai_experiments.zoning_game.zg_cfg_policy import create_policy_cfg_with_fallback

def _my_evaluate_policies_for_seed(policy_seed, ruleset, fallback_policy_creator, env_seeds, env, on_invalid):
    logging.getLogger("nsai_experiments.zoning_game.zg_cfg").setLevel(logging.ERROR)
    logging.getLogger("nsai_experiments.zoning_game.zg_gym").setLevel(logging.ERROR)

    ruleset_policy = create_policy_cfg_with_fallback(ruleset, fallback_policy_creator, seed=policy_seed)

    local_ruleset_scores = []
    local_ruleset_infos = []

    for env_seed in env_seeds:
        _, _, ruleset_reward, _, _, ruleset_info = play_one_game(ruleset_policy, env=env, seed=env_seed, on_invalid=on_invalid)
        local_ruleset_scores.append(ruleset_reward)
        local_ruleset_infos.append(ruleset_info)
    return local_ruleset_scores, local_ruleset_infos
