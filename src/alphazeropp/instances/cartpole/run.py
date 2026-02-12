import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator

from .game import CartPoleGame
from .network import CartPolePolicyValueNet


def run_single_episode(
    agent: Agent,
    game: CartPoleGame,
    seed: int | None = None,
    max_moves: int = 500,
):
    """Run one episode using the agent policy and return total reward and steps."""
    rng = np.random.default_rng(seed)
    game.reset_wrapper(seed=seed)
    total_reward = 0.0
    for _ in range(max_moves):
        move_probs = agent.policy(game)
        action = int(rng.choice(len(move_probs), p=move_probs))
        _, reward, terminated, truncated, _ = game.step_wrapper(action)
        if reward is not None:
            total_reward += float(reward)
        if terminated or truncated:
            break
    return total_reward, game.step_count


def main():
    game = CartPoleGame()
    net = CartPolePolicyValueNet()
    agent = Agent(
        game=game,
        net=net,
        mcts_params={"n_simulations": 25, "temperature": 1.0, "c_exploration": 1.0},
        external_policy=None,
    )

    trainer = Trainer(
        agent=agent,
        net=net,
        game=game,
        n_games_per_train=10,
        n_past_iterations_to_train=5,
        n_procs=-1,
    )
    evaluator = Evaluator(n_games=5, n_procs=-1)

    total_reward, steps = run_single_episode(agent, game, seed=0)
    print(f"Episode done: steps={steps}, total_reward={total_reward:.3f}")

    # Example usage:
    # trainer.train_iteration()
    # evaluator.pit(agent_new=agent, agent_old=agent, eval_seed=0, mcts_seed=1, external_policy_seed=2)


if __name__ == "__main__":
    main()
