from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import numpy as np

from alphazeropp.core.agent import Agent
from alphazeropp.training.trainer import Trainer
from alphazeropp.training.evaluator import Evaluator

from alphazeropp.instances.cartpole.game import CartPoleGame
from alphazeropp.instances.cartpole.network import CartPolePolicyValueNet

import copy

import torch

import logging

def models_equal(m1, m2):
    sd1 = m1.state_dict()
    sd2 = m2.state_dict()

    if sd1.keys() != sd2.keys():
        return False

    for k in sd1:
        if not torch.equal(sd1[k], sd2[k]):
            return False

    return True

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
    evaluator = Evaluator(n_games=20, n_procs=-1)

    # Example usage:
    for i in range(20):
        old_agent = copy.deepcopy(trainer.agent)
        trainer.train_multiple(n_iterations=1)
        new_agent = copy.deepcopy(trainer.agent)
        score = evaluator.pit(new_agent=new_agent, old_agent=old_agent)
        print(score)
        if score >= 0.55:
            print("Keeping the new network")
            trainer.net = new_agent.net
            agent.net = new_agent.net
        else:
            print("Reverting to the old network")
            trainer.net = old_agent.net
            agent.net = old_agent.net

        if i % 5 == 0:
            plot_training_metrics(trainer.statistics_manager, evaluator.statistics_manager, save_path="cartpole_training_metrics.png")

def plot_training_metrics(trainer_stats_manager, evaluator_stats_manager, save_path=None):
    """
    Sub-Phase 4.2: Plot training metrics.
    
    Justification:
        Visualization is essential for diagnosing training issues and comparing runs.
        Plots eval rewards, training losses, and training set size over iterations.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[Warning] matplotlib/pandas not installed. Skipping plot generation.")
        return

    trainer_history = trainer_stats_manager.to_list() if trainer_stats_manager else []
    evaluator_history = evaluator_stats_manager.to_list() if evaluator_stats_manager else []
    max_len = max(len(trainer_history), len(evaluator_history))
    if max_len == 0:
        print("[Warning] No statistics available for plotting.")
        return

    merged = []
    for i in range(max_len):
        row = {"iteration": i + 1}
        if i < len(trainer_history):
            row.update(trainer_history[i])
        if i < len(evaluator_history):
            row.update(evaluator_history[i])
        merged.append(row)

    df = pd.DataFrame(merged)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("CartPole AlphaZero Training Metrics", fontsize=14, fontweight='bold')
    
    # Plot 1: Eval reward mean with std band
    ax1 = axes[0, 0]
    if 'new_rewards_mean' in df.columns:
        ax1.plot(df['iteration'], df['new_rewards_mean'], 'b-', linewidth=2, label='New Reward Mean')
        if 'new_rewards_std' in df.columns:
            ax1.fill_between(
                df['iteration'],
                df['new_rewards_mean'] - df['new_rewards_std'],
                df['new_rewards_mean'] + df['new_rewards_std'],
                alpha=0.3, color='blue', label='±1 Std'
            )
    if 'old_rewards_mean' in df.columns:
        ax1.plot(df['iteration'], df['old_rewards_mean'], 'c--', linewidth=2, label='Old Reward Mean')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reward')
    ax1.set_title('Evaluation Rewards')
    if len(ax1.lines) > 0:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Policy and Value Loss
    ax2 = axes[0, 1]
    has_loss = False
    if 'train_loss_policy' in df.columns:
        ax2.plot(df['iteration'], df['train_loss_policy'], 'r-', linewidth=2, label='Policy Loss')
        has_loss = True
    if 'train_loss_value' in df.columns:
        ax2.plot(df['iteration'], df['train_loss_value'], 'g-', linewidth=2, label='Value Loss')
        has_loss = True
    if not has_loss and 'train_loss' in df.columns:
        ax2.plot(df['iteration'], df['train_loss'], 'k-', linewidth=2, label='Train Loss')
        has_loss = True
    if has_loss:
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Losses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    else:
        ax2.axis('off')
    
    # Plot 3: Training Set Size or Game Length (if available)
    ax3 = axes[1, 0]
    if 'num_examples' in df.columns:
        ax3.plot(df['iteration'], df['num_examples'], 'm-', linewidth=2, marker='o', markersize=4)
        ax3.set_ylabel('Num Examples')
        ax3.set_title('Training Set Size')
    elif 'game_length' in df.columns:
        ax3.plot(df['iteration'], df['game_length'], 'm-', linewidth=2, marker='o', markersize=4)
        ax3.set_ylabel('Avg Game Length')
        ax3.set_title('Average Episode Length')
    ax3.set_xlabel('Iteration')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined (normalized)
    ax4 = axes[1, 1]
    series_to_plot = []
    if 'new_rewards_mean' in df.columns and df['new_rewards_mean'].max() > 0:
        reward_norm = df['new_rewards_mean'] / df['new_rewards_mean'].max()
        series_to_plot.append((reward_norm, 'b-', 'Reward (norm)'))
    if 'num_examples' in df.columns and df['num_examples'].max() > 0:
        examples_norm = df['num_examples'] / df['num_examples'].max()
        series_to_plot.append((examples_norm, 'm-', 'Num Examples (norm)'))
    if 'train_loss_policy' in df.columns and df['train_loss_policy'].max() > 0:
        policy_norm = 1 - (df['train_loss_policy'] / df['train_loss_policy'].max())
        series_to_plot.append((policy_norm, 'r--', '1 - Policy Loss (norm)'))

    for series, style, label in series_to_plot:
        ax4.plot(df['iteration'], series, style, linewidth=2, label=label)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Combined Metrics (Normalized)')
    if series_to_plot:
        ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved to {save_path}")
    
    plt.close(fig)  # Close to free memory
    
    

if __name__ == "__main__":
    main()
