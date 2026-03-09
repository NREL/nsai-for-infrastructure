"""
Doors Direct Play -- AlphaZero Training.

Runs the AlphaZero pipeline with MCTS directly on the Doors environment
(no DSL/grammar). Matched hyperparameters to the grammar game for fair comparison.

Usage:
    python scripts/run_doors_direct.py
"""

from alphazeropp.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from alphazeropp.instances.doors.config import DoorsDirectConfig, compute_dims
from alphazeropp.instances.doors.eval_ablations import (
    ABLATION_MODES, compute_solve_rate_ablated,
)
from alphazeropp.training.gated_trainer import GatedTrainer
from alphazeropp.utils.interactive_config import (
    build_param_list, interactive_edit, attr_setter, dict_setter,
)


# ---------------------------------------------------------------------------
# Interactive config editing
# ---------------------------------------------------------------------------

def _build_sections(cfg, seed_state=None):
    """Return sections for Doors Direct Play config."""

    def _update_dims():
        """Recompute net sizes from current game kwargs."""
        nr = cfg.game.kwargs.get("num_rooms", 2)
        lpr = cfg.game.kwargs.get("locs_per_room", 2)
        dims = compute_dims(nr, lpr)
        cfg.net.kwargs["input_size"] = dims["obs_size"]
        cfg.net.kwargs["output_size"] = dims["action_count"]
        cfg.net.kwargs["hidden_size"] = max(64, dims["obs_size"] * 4)

    def _set_num_rooms(val):
        cfg.game.kwargs["num_rooms"] = int(val)
        _update_dims()

    def _set_locs_per_room(val):
        cfg.game.kwargs["locs_per_room"] = int(val)
        _update_dims()

    num_rooms = cfg.game.kwargs.get("num_rooms", 2)
    locs_per_room = cfg.game.kwargs.get("locs_per_room", 2)

    params = [
        ("num_rooms", num_rooms, _set_num_rooms,
         "Number of rooms (D>=2, higher=harder)", None),
        ("locs_per_room", locs_per_room, _set_locs_per_room,
         "Locations per room (more=larger action space)", None),
        ("step_penalty", cfg.game.kwargs.get("step_penalty", 0.01),
         dict_setter(cfg.game.kwargs, "step_penalty"),
         "Per-step penalty", None),
        ("unlock_bonus", cfg.game.kwargs.get("unlock_bonus", 0.1),
         dict_setter(cfg.game.kwargs, "unlock_bonus"),
         "Bonus for unlocking a room", None),
        ("use_precondition_mask", cfg.game.kwargs.get("use_precondition_mask", False),
         dict_setter(cfg.game.kwargs, "use_precondition_mask"),
         "Precondition-aware action masking", None),
    ]

    mcts_descs = {
        "n_simulations":    "MCTS rollouts per move",
        "temperature":      "Exploration temperature for action selection",
        "c_exploration":    "UCB exploration constant",
        "dirichlet_alpha":  "Dirichlet noise concentration parameter",
        "dirichlet_epsilon": "Weight of Dirichlet noise at root",
    }
    for k in ["n_simulations", "temperature", "c_exploration",
              "dirichlet_alpha", "dirichlet_epsilon"]:
        if k in cfg.agent.mcts_params:
            params.append(
                (k, cfg.agent.mcts_params[k],
                 dict_setter(cfg.agent.mcts_params, k),
                 mcts_descs[k], None))

    params.extend([
        ("reward_discount", cfg.agent.reward_discount,
         attr_setter(cfg.agent, "reward_discount"),
         "Discount factor for future rewards", None),
        ("n_games_per_train", cfg.trainer.n_games_per_train,
         attr_setter(cfg.trainer, "n_games_per_train"),
         "Self-play games per training iteration", None),
        ("n_past_iters", cfg.trainer.n_past_iterations_to_train,
         attr_setter(cfg.trainer, "n_past_iterations_to_train"),
         "Past iterations kept in training buffer", None),
        ("n_procs", cfg.trainer.n_procs,
         attr_setter(cfg.trainer, "n_procs"),
         "Parallel workers for self-play (-1=sequential)", None),
        ("eval_n_games", cfg.evaluator.n_games,
         attr_setter(cfg.evaluator, "n_games"),
         "Games to pit new vs old agent", None),
        ("eval_n_procs", cfg.evaluator.n_procs,
         attr_setter(cfg.evaluator, "n_procs"),
         "Parallel workers for evaluation", None),
        ("n_iterations", cfg.run.n_iterations,
         attr_setter(cfg.run, "n_iterations"),
         "Total training iterations", None),
        ("accept_threshold", cfg.run.accept_threshold,
         attr_setter(cfg.run, "accept_threshold"),
         "Win rate to accept new network", None),
        ("plot_every", cfg.run.plot_every,
         attr_setter(cfg.run, "plot_every"),
         "Plot metrics every N iterations", None),
    ])

    if seed_state is not None:
        params.append(
            ("seeds", seed_state["seeds_str"],
             lambda val: seed_state.__setitem__("seeds_str", val),
             "Base seeds (comma-sep for multi-seed CI)", None))

    all_params = build_param_list(params)

    env_labels = {"num_rooms", "locs_per_room", "step_penalty",
                  "unlock_bonus", "use_precondition_mask"}
    mcts_labels = {"n_simulations", "temperature", "c_exploration",
                   "dirichlet_alpha", "dirichlet_epsilon"}
    agent_labels = {"reward_discount"}
    trainer_labels = {"n_games_per_train", "n_past_iters", "n_procs"}
    evaluator_labels = {"eval_n_games", "eval_n_procs"}
    run_labels = {"n_iterations", "accept_threshold", "plot_every", "seeds"}

    return [
        ("Environment", [p for p in all_params if p[1] in env_labels]),
        ("MCTS",        [p for p in all_params if p[1] in mcts_labels]),
        ("Agent",       [p for p in all_params if p[1] in agent_labels]),
        ("Trainer",     [p for p in all_params if p[1] in trainer_labels]),
        ("Evaluator",   [p for p in all_params if p[1] in evaluator_labels]),
        ("Run",         [p for p in all_params if p[1] in run_labels]),
    ]


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

def setup_experiment_dir(cfg, eval_ablation=None):
    """Create and return an experiment directory. Updates cfg paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_rooms = cfg.game.kwargs.get("num_rooms", 2)
    locs_per_room = cfg.game.kwargs.get("locs_per_room", 2)
    sim = cfg.agent.mcts_params.get("n_simulations", 0)
    games = cfg.trainer.n_games_per_train
    iters = cfg.run.n_iterations
    mask = "mask" if cfg.game.kwargs.get("use_precondition_mask", False) else "nomask"
    dirname = (f"{timestamp}_D{num_rooms}_L{locs_per_room}_{mask}"
               f"_mcts{sim}_games{games}_iter{iters}")
    if eval_ablation and eval_ablation != "full":
        dirname += f"_abl-{eval_ablation}"

    exp_dir = Path("experiments") / "doors_direct" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")

    return exp_dir


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------

def _parse_obs(obs, env):
    """Extract agent location, room lock status, key availability from obs."""
    agent_loc = int(np.argmax(obs[:env.M]))
    unlocked = [bool(obs[env._unlocked_offset + r]) for r in range(env.D)]
    keys_available = [bool(obs[env._key_offset + k]) for k in range(env.K)]
    return agent_loc, unlocked, keys_available


def _problem_description(num_rooms, locs_per_room=2):
    """Return a human-readable problem description."""
    M = num_rooms * locs_per_room
    K = num_rooms - 1
    optimal_steps = 2 * K + 1
    opt_reward = _optimal_reward(num_rooms)
    return (
        f"Navigate from loc 0 to loc {M-1} through {num_rooms} rooms.\n"
        f"           Room 0 starts unlocked. {K} key(s), sequential unlock chain.\n"
        f"           Optimal plan: {optimal_steps} steps, reward {opt_reward:+.2f}"
    )


def _optimal_reward(num_rooms, step_penalty=0.01, unlock_bonus=0.1):
    """Compute optimal reward for any D (with default penalty/bonus)."""
    optimal_steps = 2 * (num_rooms - 1) + 1
    return 1.0 + (num_rooms - 1) * unlock_bonus - optimal_steps * step_penalty


def _print_room_layout(env, num_rooms, locs_per_room):
    """Print an ASCII diagram of the room/key/goal layout."""
    # Build per-room content
    key_at_loc = {}
    for k in range(env.K):
        key_at_loc[env.key_loc[k]] = k

    cols_per_row = min(num_rooms, 4)
    # Compute width needed: "loc XX" is 6 chars, annotations "[key XX]" is 8
    max_loc = num_rooms * locs_per_room - 1
    loc_col_width = max(len(f"loc {max_loc}"), 8) + 2  # padding
    room_width = max(loc_col_width * locs_per_room + 2, 22)

    for row_start in range(0, num_rooms, cols_per_row):
        row_end = min(row_start + cols_per_row, num_rooms)
        rooms_in_row = range(row_start, row_end)

        # Room header line
        header = "    "
        for r in rooms_in_row:
            status = "UNLOCKED" if r == 0 else "LOCKED"
            cell = f" Room {r} ({status})"
            header += cell.ljust(room_width) + "|"
        print(header.rstrip("|"))

        # Location line
        loc_line = "    "
        for r in rooms_in_row:
            locs = [l for l in range(env.M) if env.loc_room[l] == r]
            cell = " " + "  ".join(f"loc {l:<2}" for l in locs)
            loc_line += cell.ljust(room_width) + "|"
        print(loc_line.rstrip("|"))

        # Annotation line (start, keys, goal)
        ann_line = "    "
        for r in rooms_in_row:
            locs = [l for l in range(env.M) if env.loc_room[l] == r]
            annotations = []
            for l in locs:
                if l == env.start_loc and l == env.goal_loc:
                    annotations.append("[S,G]")
                elif l == env.start_loc:
                    annotations.append("[start]")
                elif l == env.goal_loc:
                    annotations.append("[goal]")
                elif l in key_at_loc:
                    annotations.append(f"[key {key_at_loc[l]}]")
                else:
                    annotations.append("")
            cell = " " + "  ".join(f"{a:<6}" for a in annotations)
            ann_line += cell.ljust(room_width) + "|"
        print(ann_line.rstrip("|"))
        print()


def print_banner(cfg, game, exp_dir):
    """Print startup banner with environment info and hyperparameters."""
    env = game.env
    num_rooms = cfg.game.kwargs.get("num_rooms", 2)
    locs_per_room = cfg.game.kwargs.get("locs_per_room", 2)
    dims = compute_dims(num_rooms, locs_per_room)
    sep = "=" * 80

    print()
    print(sep)
    print(f"  Doors D={num_rooms} Direct Play -- AlphaZero Training")
    print(sep)
    print()
    print(f"  PROBLEM (D={num_rooms}, {locs_per_room} locs/room)")
    print()

    # Room layout diagram
    _print_room_layout(env, num_rooms, locs_per_room)
    print()

    # Key dependency chain
    print("  KEY DEPENDENCIES (sequential chain)")
    for k in range(env.K):
        key_room = env.loc_room[env.key_loc[k]]
        print(f"    key {k} (loc {env.key_loc[k]}, room {key_room})"
              f" --- unlocks room {env.key_unlocks[k]}")
    print()

    # Optimal plan
    K = env.K
    optimal_steps = 2 * K + 1
    opt_actions = []
    for k in range(K):
        opt_actions.append(f"MOVE_TO({env.key_loc[k]})")
        opt_actions.append(f"PICK({k})")
    opt_actions.append(f"MOVE_TO({env.goal_loc})")
    print(f"  OPTIMAL PLAN ({optimal_steps} steps)")
    # Print in rows of ~5 actions
    for row_start in range(0, len(opt_actions), 5):
        chunk = opt_actions[row_start:row_start + 5]
        line = " -> ".join(chunk)
        prefix = "    " if row_start == 0 else "    -> "
        print(f"{prefix}{line}")
    print()

    # Reward breakdown
    opt_reward = _optimal_reward(num_rooms, env.step_penalty, env.unlock_bonus)
    print(f"  REWARDS")
    print(f"    goal         = +1.00")
    print(f"    unlock_bonus = +{env.unlock_bonus:.2f}/key"
          f"  ({K} keys = +{K * env.unlock_bonus:.2f})")
    print(f"    step_penalty = -{env.step_penalty:.2f}/step"
          f"  ({optimal_steps} steps = -{optimal_steps * env.step_penalty:.2f})")
    print(f"    optimal      = +1.00 + {K}x{env.unlock_bonus:.2f}"
          f" - {optimal_steps}x{env.step_penalty:.2f} = {opt_reward:+.2f}")
    print()

    # Dimensions summary
    print(f"  DIMENSIONS")
    print(f"    Obs size:     {dims['obs_size']}"
          f"  ({env.M} locs + {env.D} rooms + {env.K} keys)")
    print(f"    Action space: {dims['action_count']}"
          f"  ({env.M} MOVE_TO + {env.K} PICK + 1 NOOP)")
    print(f"    Horizon:      {env.horizon}")
    print()

    # Action semantics
    print("  ACTIONS")
    print("    MOVE_TO(l)  Teleport agent to location l. If the room containing l")
    print("                is locked, the action is silently ignored (no-op).")
    print("    PICK(k)     Pick up key k. Requires agent at key_loc[k] and key k")
    print("                available. Unlocks the target room immediately.")
    print("    NOOP        Do nothing. Always valid.")
    print()

    # Training hyperparameters
    print("  TRAINING")
    mp = cfg.agent.mcts_params
    print(f"    MCTS simulations:    {mp['n_simulations']}")
    print(f"    Exploration (c):     {mp['c_exploration']}")
    print(f"    Dirichlet:           alpha={mp['dirichlet_alpha']},"
          f" eps={mp['dirichlet_epsilon']}")
    print(f"    Reward discount:     {cfg.agent.reward_discount}")
    print(f"    Games per iteration: {cfg.trainer.n_games_per_train}")
    print(f"    Training iterations: {cfg.run.n_iterations}")
    print(f"    Gate threshold:      {cfg.run.accept_threshold}")
    use_mask = cfg.game.kwargs.get('use_precondition_mask', False)
    print(f"    Precondition mask:   {use_mask}")
    print()

    # Explain precondition masking
    print("  ACTION MASKING")
    if use_mask:
        print("    Mode: PRECONDITION-AWARE (enabled)")
        print("    Only actions whose PDDL preconditions are satisfied are valid.")
        print("    MOVE_TO(l): valid iff the room containing loc l is unlocked.")
        print("    PICK(k):    valid iff agent is at key_loc[k] AND key k is available.")
        print("    NOOP:       always valid.")
        print("    This prunes the search tree but gives the agent domain knowledge.")
    else:
        print("    Mode: STRUCTURAL (default)")
        print("    All semantic actions (MOVE_TO, PICK, NOOP) are always valid.")
        print("    Invalid moves (e.g., MOVE_TO a locked room) are silently")
        print("    ignored by the env (no-op). The agent must learn preconditions")
        print("    from experience alone.")
    print()

    # Output legend
    print("  Output legend:")
    print("    [TRAIN]   Self-play data collection & network training")
    print("    [EVAL]    Pitting new network vs old network")
    print("    [ITER]    Iteration summary with key metrics")
    print("    [POLICY]  Greedy policy trace (what the agent learned)")
    print()
    print(f"  Experiment dir: {exp_dir}/")
    print(sep)
    print()


def print_architecture(cfg):
    """Print the AlphaZero training loop architecture diagram."""
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    n_past = cfg.trainer.n_past_iterations_to_train
    num_rooms = cfg.game.kwargs.get("num_rooms", 2)
    locs_per_room = cfg.game.kwargs.get("locs_per_room", 2)
    dims = compute_dims(num_rooms, locs_per_room)
    obs_size = dims["obs_size"]
    horizon = dims["horizon"]

    print("=" * 80)
    print("  Algorithm Architecture: AlphaZero (Self-Play + Learning)")
    print("=" * 80)
    print()
    print(f"  Game: Doors with {num_rooms} rooms. State is a {obs_size}-dim"
          f" vector of locations, locks, and keys.")
    print(f"  Outer loop: {n_iters} training iterations")
    print(f"  Each iteration: {n_games} self-play games -> train net -> evaluate")
    print()
    print("  +-- Iteration i "
          "------------------------------------------------------+")
    print("  |                                                    "
          "                  |")
    print(f"  |  STEP 1: Self-Play  ({n_games} games, each up to"
          f" {horizon} steps)")
    print("  |")
    print("  |    Game:  s0 --action--> s1 --action--> s2 --> ... --> terminal")
    print("  |")
    print("  |    Each move creates a FRESH MCTS tree:")
    print(f"  |      mcts = MCTS(state, neural_net, n_sims={n_sims})")
    print("  |      +----------------------------------------------------+")
    print(f"  |      |  for sim in 1..{n_sims}:")
    print("  |      |    SELECT  -> traverse tree via UCB")
    print("  |      |              UCB = Q_norm(s,a) + c * pi_net(a)"
          " * sqrt(N)/(1+Na)")
    print("  |      |    EXPAND  -> hit new node, query net f(s) -> (pi, v)")
    print("  |      |    BACKUP  -> Q(s,a) = running_mean(R(s,a) + V(s'))")
    print("  |      +----------------------------------------------------+")
    print("  |      pi_MCTS = visit_counts^(1/tau) / sum")
    print("  |      action ~ pi_MCTS                        <- sample move")
    print("  |      SAVE (state, pi_MCTS)                   <- training target")
    print("  |      DISCARD tree                             <- net carries knowledge")
    print("  |")
    print(f"  |  STEP 2: Train MLP  (replay buffer: last {n_past} iterations)")
    print("  |")
    print("  |    L = (z - v(s))^2  -  pi_MCTS * log p(s)  +  c*||theta||^2")
    print("  |        value loss       policy loss              weight decay")
    print("  |    theta <- theta - alpha * grad(L)")
    print("  |")
    print("  |  STEP 3: Evaluate")
    print("  |    Pit new_net vs old_net. Accept if win_rate >= threshold.")
    print("  |                                                    "
          "                  |")
    print("  +----------------------------------------------------"
          "------------------+")
    print()
    print("  Knowledge flow:")
    print("    Trees are DISPOSABLE. They produce training data (s, pi_MCTS, z).")
    print("    Neural net f_theta is the KNOWLEDGE CARRIER across iterations.")
    print("    Better net -> better MCTS -> better data -> better net"
          " (virtuous cycle)")
    print()


def print_iteration_header(i, total):
    """Print iteration separator."""
    header = f"--- Iteration {i}/{total} "
    print(header + "-" * (80 - len(header)))


def print_iteration_summary(i, total, score, accepted, solve_rate, best_solve_rate,
                            trainer_stats, evaluator_stats, elapsed,
                            avg_eval_reward=None):
    """Print compact one-line iteration summary."""
    train_rec = (trainer_stats.to_list()[-1]
                 if trainer_stats and trainer_stats.to_list() else {})
    eval_rec = (evaluator_stats.to_list()[-1]
                if evaluator_stats and evaluator_stats.to_list() else {})

    loss = train_rec.get("train_loss", float("nan"))
    p_loss = train_rec.get("train_loss_policy", float("nan"))
    v_loss = train_rec.get("train_loss_value", float("nan"))
    n_examples = train_rec.get("num_examples", 0)

    if evaluator_stats is not None:
        new_mean = eval_rec.get("new_rewards_mean", float("nan"))
        old_mean = eval_rec.get("old_rewards_mean", float("nan"))
        reward_str = f"New: {new_mean:+.3f} Old: {old_mean:+.3f}"
    else:
        reward_str = f"Eval: {avg_eval_reward:+.3f}" if avg_eval_reward is not None else ""

    marker = "ACCEPT" if accepted else "REJECT"
    print(
        f"[ITER {i}/{total}] Score: {score*100:.1f}% [{marker}]"
        f" | solve={solve_rate:.2f} best={best_solve_rate:.2f}"
        f" | {reward_str}"
        f" | Loss: {loss:.4f} (P:{p_loss:.3f} V:{v_loss:.3f})"
        f" | {n_examples} examples"
        f" | {elapsed:.1f}s"
    )


def play_greedy_episode(agent):
    """Play one greedy episode and return (steps, total_reward, solved, trace).

    trace is a list of (obs_before, action_idx, action_name, reward, probs).
    """
    game = agent.game.clone()
    game.reset_wrapper(seed=0)
    action_specs = game.env.action_spec
    action_names = {a.index: a.name for a in action_specs}

    trace = []
    total_reward = 0.0

    for _ in range(game.env.horizon):
        obs_before = game.obs.copy()
        probs = agent.policy(game, "", add_noise=False,
                             temperature_override=0.01)
        action = int(np.argmax(probs))
        _, reward, terminated, truncated, _ = game.step_wrapper(action)
        total_reward += reward

        trace.append((obs_before, action,
                       action_names.get(action, f"?({action})"),
                       reward, probs))

        if terminated or truncated:
            break

    solved = game.env.is_solved(game.obs)
    return game.step_count, total_reward, solved, trace, game.obs.copy()


def print_learnt_policy(agent):
    """Print the current greedy policy as a compact one-liner."""
    steps, total_reward, solved, trace, _ = play_greedy_episode(agent)
    action_seq = " -> ".join(name for _, _, name, _, _ in trace)
    status = "SOLVED" if solved else "NOT SOLVED"
    print(f"  [POLICY] {action_seq} -> {status}"
          f" ({steps} steps, reward={total_reward:+.3f})")
    print()


def compute_solve_rate(agent, n_episodes=20):
    """Play episodes greedily and compute solve rate + avg reward."""
    solves = 0
    total_reward = 0.0

    for i in range(n_episodes):
        game = agent.game.clone()
        game.reset_wrapper(seed=1000 + i)
        cumulative = 0.0

        for _ in range(game.env.horizon):
            probs = agent.policy(game, "", add_noise=False,
                                 temperature_override=0.01)
            action = int(np.argmax(probs))
            _, reward, terminated, truncated, _ = game.step_wrapper(action)
            cumulative += reward
            if terminated or truncated:
                break

        if game.env.is_solved(game.obs):
            solves += 1
        total_reward += cumulative

    return solves / n_episodes, total_reward / n_episodes


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_metrics(trainer_stats_manager, evaluator_stats_manager,
                          iteration_log, save_path=None, optimal_reward=None,
                          num_rooms=2, title=None):
    """Plot training metrics: rewards, losses, solve rate, combined."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[Warning] matplotlib/pandas not installed. Skipping plot.")
        return

    # Merge trainer + evaluator stats by iteration index
    trainer_history = (trainer_stats_manager.to_list()
                       if trainer_stats_manager else [])
    evaluator_history = (evaluator_stats_manager.to_list()
                         if evaluator_stats_manager else [])
    max_len = max(len(trainer_history), len(evaluator_history))
    if max_len == 0:
        return

    merged = []
    for i in range(max_len):
        row = {"iteration": i + 1}
        if i < len(trainer_history):
            row.update(trainer_history[i])
        if i < len(evaluator_history):
            row.update(evaluator_history[i])
        # Add solve rate from iteration_log
        if i < len(iteration_log):
            row["solve_rate"] = iteration_log[i].get("solve_rate", 0)
            row["best_solve_rate"] = iteration_log[i].get("best_solve_rate", 0)
            row["avg_eval_reward"] = iteration_log[i].get("avg_eval_reward", 0)
        merged.append(row)

    df = pd.DataFrame(merged)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title or f"Doors D={num_rooms} Direct Play -- Training Metrics",
                 fontsize=14, fontweight="bold")

    # Plot 1: Evaluation Rewards
    ax1 = axes[0, 0]
    if optimal_reward is not None:
        ax1.axhline(y=optimal_reward, color="green", linestyle="--",
                    linewidth=1.5, label=f"Optimal ({optimal_reward:.2f})")
    if "new_rewards_mean" in df.columns:
        ax1.plot(df["iteration"], df["new_rewards_mean"], "b-",
                 linewidth=2, label="New Reward Mean")
        if "new_rewards_std" in df.columns:
            upper = df["new_rewards_mean"] + df["new_rewards_std"]
            lower = df["new_rewards_mean"] - df["new_rewards_std"]
            if optimal_reward is not None:
                upper = np.minimum(upper, optimal_reward)
            ax1.fill_between(df["iteration"], lower, upper,
                             alpha=0.3, color="blue", label="+-1 Std")
        if "old_rewards_mean" in df.columns:
            ax1.plot(df["iteration"], df["old_rewards_mean"], "c--",
                     linewidth=2, label="Old Reward Mean")
    elif "avg_eval_reward" in df.columns:
        ax1.plot(df["iteration"], df["avg_eval_reward"], "b-",
                 linewidth=2, label="Eval Reward (greedy)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reward")
    ax1.set_title("Evaluation Rewards")
    if len(ax1.lines) > 0:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Losses
    ax2 = axes[0, 1]
    has_loss = False
    if "train_loss_policy" in df.columns:
        ax2.plot(df["iteration"], df["train_loss_policy"], "r-",
                 linewidth=2, label="Policy Loss")
        has_loss = True
    if "train_loss_value" in df.columns:
        ax2.plot(df["iteration"], df["train_loss_value"], "g-",
                 linewidth=2, label="Value Loss")
        has_loss = True
    if not has_loss and "train_loss" in df.columns:
        ax2.plot(df["iteration"], df["train_loss"], "k-",
                 linewidth=2, label="Train Loss")
        has_loss = True
    if has_loss:
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Losses")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")
    else:
        ax2.axis("off")

    # Plot 3: Solve Rate
    ax3 = axes[1, 0]
    if "solve_rate" in df.columns:
        ax3.plot(df["iteration"], df["solve_rate"], "g-", linewidth=2,
                 marker="o", markersize=4, label="Solve Rate")
    if "best_solve_rate" in df.columns:
        ax3.plot(df["iteration"], df["best_solve_rate"], "g--",
                 linewidth=1.5, alpha=0.6, label="Best Solve Rate")
    ax3.axhline(y=1.0, color="green", linestyle=":", linewidth=1,
                alpha=0.4)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Solve Rate")
    ax3.set_title("Solve Rate Over Training")
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Combined Normalized
    ax4 = axes[1, 1]
    series_to_plot = []
    if "new_rewards_mean" in df.columns and df["new_rewards_mean"].max() > 0:
        reward_norm = df["new_rewards_mean"] / df["new_rewards_mean"].max()
        series_to_plot.append((reward_norm, "b-", "Reward (norm)"))
    elif "avg_eval_reward" in df.columns and df["avg_eval_reward"].max() > 0:
        reward_norm = df["avg_eval_reward"] / df["avg_eval_reward"].max()
        series_to_plot.append((reward_norm, "b-", "Reward (norm)"))
    if "solve_rate" in df.columns:
        series_to_plot.append((df["solve_rate"], "g-", "Solve Rate"))
    if ("train_loss_policy" in df.columns
            and df["train_loss_policy"].max() > 0):
        policy_norm = 1 - (df["train_loss_policy"]
                           / df["train_loss_policy"].max())
        series_to_plot.append((policy_norm, "r--", "1 - Policy Loss (norm)"))

    for series, style, label in series_to_plot:
        ax4.plot(df["iteration"], series, style, linewidth=2, label=label)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Normalized Value")
    ax4.set_title("Combined Metrics (Normalized)")
    if series_to_plot:
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved to {save_path}")

    plt.close(fig)


def rename_plot_with_stats(cfg, iteration_log):
    """Rename the plot file to include key stats in the filename."""
    plot_path = Path(cfg.run.plot_path)
    if not plot_path.exists():
        return str(plot_path)

    num_rooms = cfg.game.kwargs.get("num_rooms", 2)
    best_sr = (iteration_log[-1].get("best_solve_rate", 0)
               if iteration_log else 0)
    sr_str = f"sr{best_sr*100:.0f}pct"
    avg_reward = (iteration_log[-1].get("avg_eval_reward", 0)
                  if iteration_log else 0)
    rw_str = f"reward{avg_reward:.2f}".replace(".", "p").replace("-", "m")

    new_name = plot_path.with_name(
        f"metrics_D{num_rooms}_{sr_str}_{rw_str}.png")
    plot_path.rename(new_name)
    return str(new_name)


def render_policy_gif(trace, final_obs, env, save_path, fps=2):
    """Render an animated GIF of the greedy policy trace.

    Parameters
    ----------
    trace : list of (obs_before, action_idx, action_name, reward, probs)
    final_obs : np.ndarray – observation after the last action.
    env : DoorsPDDLLiteEnv – provides layout metadata.
    save_path : str or Path – output GIF file path.
    fps : int – frames per second for action frames (default 2).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from io import BytesIO
        from PIL import Image
    except ImportError:
        print("[Warning] matplotlib or Pillow not installed. Skipping GIF.")
        return

    if not trace:
        print("[Warning] Empty trace, skipping GIF generation.")
        return

    # --- Build per-frame data: (obs, label) ---
    frames_data = []
    frames_data.append((trace[0][0], "Initial State  (total=+0.000)"))
    cumulative_reward = 0.0
    for i, (_, _, action_name, reward, _) in enumerate(trace):
        cumulative_reward += reward
        obs_after = trace[i + 1][0] if i + 1 < len(trace) else final_obs
        label = f"Step {i + 1}/{len(trace)}: {action_name}  (total={cumulative_reward:+.3f})"
        frames_data.append((obs_after, label))

    # --- Layout geometry ---
    locs_per_room_counts = {}
    for loc in range(env.M):
        r = env.loc_room[loc]
        locs_per_room_counts[r] = locs_per_room_counts.get(r, 0) + 1
    max_locs_in_room = max(locs_per_room_counts.values())

    room_w = max(2.0, 0.8 * max_locs_in_room)
    room_h = 1.5
    gap = 0.6
    cols = min(env.D, 5)
    row_count = (env.D + cols - 1) // cols

    fig_w = cols * (room_w + gap) - gap + 1.0
    fig_h = row_count * (room_h + gap) - gap + 2.0

    # Room centres
    room_pos = {}
    for r in range(env.D):
        c = r % cols
        rw = r // cols
        cx = 0.5 + c * (room_w + gap) + room_w / 2
        cy = fig_h - 1.5 - rw * (room_h + gap) - room_h / 2
        room_pos[r] = (cx, cy)

    # Location centres within rooms
    loc_pos = {}
    for r in range(env.D):
        locs = [loc for loc in range(env.M) if env.loc_room[loc] == r]
        cx, cy = room_pos[r]
        n = len(locs)
        for j, loc in enumerate(locs):
            lx = cx - room_w / 2 + room_w * (j + 0.5) / n
            loc_pos[loc] = (lx, cy)

    # Key-at-location map
    key_at_loc = {}
    for k in range(env.K):
        key_at_loc[env.key_loc[k]] = k

    # --- Render frames ---
    pil_frames = []
    for obs, label in frames_data:
        agent_loc, unlocked, keys_avail = _parse_obs(obs, env)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(label, fontsize=13, fontweight="bold", pad=12)

        # Draw rooms
        for r in range(env.D):
            cx, cy = room_pos[r]
            bg = "#c8e6c9" if unlocked[r] else "#ffcdd2"
            ec = "#2e7d32" if unlocked[r] else "#c62828"
            lw = 2.5 if unlocked[r] else 2.0
            rect = mpatches.FancyBboxPatch(
                (cx - room_w / 2, cy - room_h / 2), room_w, room_h,
                boxstyle="round,pad=0.08",
                facecolor=bg, edgecolor=ec, linewidth=lw,
            )
            ax.add_patch(rect)
            lock_txt = "" if unlocked[r] else " [LOCKED]"
            ax.text(cx, cy + room_h / 2 - 0.12,
                    f"Room {r}{lock_txt}",
                    ha="center", va="top", fontsize=9, fontweight="bold",
                    color=ec)

        # Draw locations, agent, keys, goal, start
        for loc in range(env.M):
            lx, ly = loc_pos[loc]
            is_agent = (loc == agent_loc)
            is_goal = (loc == env.goal_loc)
            is_start = (loc == env.start_loc)

            if is_agent:
                circ = plt.Circle((lx, ly), 0.20, color="#1565c0", zorder=5)
                ax.add_patch(circ)
                ax.text(lx, ly, "A", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold",
                        zorder=6)
            else:
                circ = plt.Circle((lx, ly), 0.15, color="#e0e0e0",
                                  edgecolor="#757575", linewidth=1, zorder=3)
                ax.add_patch(circ)

            if is_goal:
                ax.plot(lx, ly - 0.30, "*", color="#ffd600",
                        markersize=14, zorder=4)
                ax.text(lx, ly - 0.48, "GOAL", ha="center", va="top",
                        fontsize=6, color="#f57f17", fontweight="bold")

            if is_start and not is_agent:
                ax.text(lx, ly + 0.28, "S", ha="center", va="bottom",
                        fontsize=8, color="#0d47a1", fontweight="bold")

            if loc in key_at_loc:
                k = key_at_loc[loc]
                if keys_avail[k]:
                    ax.plot(lx + 0.22, ly + 0.12, "D",
                            color="#ff8f00", markersize=9, zorder=4)
                    ax.text(lx + 0.22, ly + 0.32, f"K{k}",
                            ha="center", va="bottom", fontsize=6,
                            color="#e65100", fontweight="bold")

            # Location label
            y_off = -0.48 if is_goal else -0.28
            ax.text(lx, ly + y_off, f"L{loc}", ha="center", va="top",
                    fontsize=6, color="#616161")

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()

    # --- Assemble GIF ---
    base_ms = 1000 // fps
    if len(pil_frames) == 1:
        durations = [base_ms * 3]
    else:
        durations = ([base_ms * 2]
                     + [base_ms] * (len(pil_frames) - 2)
                     + [base_ms * 3])

    pil_frames[0].save(
        str(save_path), save_all=True,
        append_images=pil_frames[1:],
        duration=durations, loop=0,
    )
    print(f"[GIF] Saved policy animation to {save_path}")


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _load_iteration_logs(seed_dirs):
    """Load iteration_log.jsonl from each seed directory."""
    logs = {}
    for seed, seed_dir in seed_dirs.items():
        path = seed_dir / "iteration_log.jsonl"
        if not path.exists():
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            logs[seed] = records
    return logs


def _load_train_stats(seed_dirs):
    """Load train_stats.jsonl from each seed directory."""
    stats = {}
    for seed, seed_dir in seed_dirs.items():
        path = seed_dir / "train_stats.jsonl"
        if not path.exists():
            continue
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            stats[seed] = records
    return stats


def _print_seed_summary(logs, seeds):
    """Print aggregated summary table across seeds."""
    if not logs:
        return

    final_solve = []
    final_reward = []
    for seed in seeds:
        if seed not in logs or not logs[seed]:
            continue
        last = logs[seed][-1]
        final_solve.append(last.get("best_solve_rate", 0))
        final_reward.append(last.get("avg_eval_reward", 0))

    n = len(final_solve)
    if n == 0:
        return

    print()
    print("=" * 70)
    print(f"  MULTI-SEED SUMMARY  ({n} seeds: {seeds})")
    print("=" * 70)
    print(f"  Solve rate:    {np.mean(final_solve):.2f} +/- {np.std(final_solve):.2f}")
    print(f"  Avg reward:    {np.mean(final_reward):+.3f} +/- {np.std(final_reward):.3f}")
    print()
    print(f"  {'Seed':>6}  {'Solve':>6}  {'Reward':>8}")
    print(f"  {'---':>6}  {'---':>6}  {'---':>8}")
    for i, seed in enumerate(seeds):
        if seed not in logs:
            continue
        print(f"  {seed:>6}  {final_solve[i]:>5.0%}  "
              f"{final_reward[i]:>+8.3f}")
    print("=" * 70)


def _sausage_panel(ax, seeds, data_dict, extractor, ylabel,
                   color="#1f77b4", label=None, ylim=None, yscale=None,
                   hline=None, overlay=False):
    """Plot mean line + std band + individual seed traces on ax."""
    if isinstance(extractor, str):
        key = extractor
        extractor = lambda r, _k=key: r.get(_k, 0)

    per_seed = []
    for seed in seeds:
        if seed not in data_dict:
            continue
        values = [extractor(e) for e in data_dict[seed]]
        per_seed.append(values)

    if not per_seed:
        return

    min_len = min(len(v) for v in per_seed)
    arr = np.array([v[:min_len] for v in per_seed])
    iters = np.arange(1, min_len + 1)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)

    lbl = label or ylabel
    ax.plot(iters, means, color=color, linewidth=2, label=lbl)
    if len(per_seed) > 1:
        ax.fill_between(iters, means - stds, means + stds,
                        color=color, alpha=0.15)
    for vals in per_seed:
        ax.plot(np.arange(1, min_len + 1), vals[:min_len],
                color=color, alpha=0.2, linewidth=0.8)

    if not overlay:
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    if yscale:
        ax.set_yscale(yscale)
    if hline:
        ax.axhline(y=hline[0], color=hline[2], linestyle=":",
                    alpha=0.5, label=hline[1])
    ax.legend(fontsize=7, loc="best")


def _plot_seed_comparison(logs, seeds, exp_dir, optimal_reward=None,
                          train_stats=None):
    """Plot 2x3 grid of sausage plots across seeds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    # (0,0) Best Solve Rate
    _sausage_panel(axes[0, 0], seeds, logs, "best_solve_rate",
                   "Best Solve Rate", color="#1f77b4",
                   ylim=(-0.05, 1.15))
    axes[0, 0].set_title("Best Solve Rate")

    # (0,1) Avg Eval Reward
    hline_arg = None
    if optimal_reward is not None:
        hline_arg = (optimal_reward, f"Optimal ({optimal_reward:.2f})", "green")
    _sausage_panel(axes[0, 1], seeds, logs, "avg_eval_reward",
                   "Avg Eval Reward", color="#1f77b4", hline=hline_arg)
    axes[0, 1].set_title("Avg Eval Reward")

    # (0,2) Training Loss
    if train_stats:
        _sausage_panel(axes[0, 2], seeds, train_stats, "train_loss_policy",
                       "Loss", color="#d62728", label="Policy Loss")
        _sausage_panel(axes[0, 2], seeds, train_stats, "train_loss_value",
                       "Loss", color="#2ca02c", label="Value Loss",
                       overlay=True, yscale="log")
        axes[0, 2].set_title("Training Loss")
    else:
        axes[0, 2].text(0.5, 0.5, "No train_stats data",
                        transform=axes[0, 2].transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
        axes[0, 2].set_title("Training Loss")

    # (1,0) Solve Rate
    _sausage_panel(axes[1, 0], seeds, logs, "solve_rate",
                   "Solve Rate", color="#9467bd")
    axes[1, 0].set_title("Solve Rate")

    # (1,1) Wall Clock per Iteration
    _sausage_panel(axes[1, 1], seeds, logs, "wall_clock_s",
                   "Wall Clock (s)", color="#8c564b")
    axes[1, 1].set_title("Wall Clock / Iteration")

    # (1,2) Self-Play Avg Reward
    if train_stats:
        _sausage_panel(axes[1, 2], seeds, train_stats, "avg_reward",
                       "Self-Play Avg Reward", color="#ff7f0e")
        axes[1, 2].set_title("Self-Play Avg Reward")
    else:
        axes[1, 2].text(0.5, 0.5, "No train_stats data",
                        transform=axes[1, 2].transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
        axes[1, 2].set_title("Self-Play Avg Reward")

    fig.suptitle(f"Seed Comparison ({len(seeds)} seeds)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = exp_dir / "seed_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Seed comparison saved to {save_path}")


# ---------------------------------------------------------------------------
# Single-seed training run
# ---------------------------------------------------------------------------

def _run_single_seed(cfg, exp_dir, eval_ablation=None):
    """Run a single-seed training loop. Returns iteration_log."""
    num_rooms = cfg.game.kwargs.get("num_rooms", 2)
    game, net, agent, trainer, evaluator = cfg.build()
    use_gating = cfg.run.accept_threshold > 0
    gated = GatedTrainer(trainer, evaluator, cfg.run.accept_threshold) if use_gating else None

    iteration_log = []
    best_solve_rate = 0.0
    best_solve_iter = 0
    n_accepted = 0
    optimal = _optimal_reward(num_rooms,
                              cfg.game.kwargs.get("step_penalty", 0.01),
                              cfg.game.kwargs.get("unlock_bonus", 0.1))
    n_sims = cfg.agent.mcts_params.get("n_simulations", "?")
    n_games = cfg.trainer.n_games_per_train
    n_iters = cfg.run.n_iterations
    plot_title = (f"Doors D={num_rooms} Direct Play\n"
                  f"sims={n_sims} games={n_games} iters={n_iters}")

    for i in range(cfg.run.n_iterations):
        t0 = time.time()
        print_iteration_header(i + 1, cfg.run.n_iterations)

        if use_gating:
            score, accepted = gated.train_iteration()
        else:
            trainer.train_iteration()
            score, accepted = 1.0, True
        elapsed = time.time() - t0

        if accepted:
            n_accepted += 1

        solve_rate, avg_eval_reward = compute_solve_rate(agent, n_episodes=20)
        if solve_rate > best_solve_rate:
            best_solve_rate = solve_rate
            best_solve_iter = i + 1

        # Ablation evaluation (if requested)
        abl_solve_rate = None
        abl_avg_reward = None
        if eval_ablation and eval_ablation != "full":
            gamma = getattr(cfg.agent, 'reward_discount', 1.0)
            abl_solve_rate, abl_avg_reward = compute_solve_rate_ablated(
                agent, eval_ablation, n_episodes=20, gamma=gamma)

        train_records = trainer.statistics_manager.to_list()
        avg_train_reward = 0.0
        if train_records:
            avg_train_reward = train_records[-1].get("avg_reward", 0.0)

        entry = {
            "iteration": i + 1,
            "gate_score": float(score),
            "accepted": bool(accepted),
            "solve_rate": solve_rate,
            "avg_train_reward": avg_train_reward,
            "avg_eval_reward": avg_eval_reward,
            "best_solve_rate": best_solve_rate,
            "wall_clock_s": round(elapsed, 1),
        }
        if abl_solve_rate is not None:
            entry["abl_solve_rate"] = abl_solve_rate
            entry["abl_avg_reward"] = abl_avg_reward
            entry["eval_ablation"] = eval_ablation
        iteration_log.append(entry)

        print_iteration_summary(
            i + 1, cfg.run.n_iterations, score, accepted,
            solve_rate, best_solve_rate,
            trainer.statistics_manager,
            evaluator.statistics_manager if use_gating else None,
            elapsed,
            avg_eval_reward=avg_eval_reward,
        )

        # Save logs
        trainer.statistics_manager.save_jsonl(
            str(exp_dir / "train_stats.jsonl"))
        if use_gating:
            evaluator.statistics_manager.save_jsonl(
                str(exp_dir / "eval_stats.jsonl"))
        with open(exp_dir / "iteration_log.jsonl", "w") as f:
            for e in iteration_log:
                f.write(json.dumps(e) + "\n")

        # Plot periodically
        eval_stats = evaluator.statistics_manager if use_gating else None
        if (i + 1) % cfg.run.plot_every == 0:
            plot_training_metrics(
                trainer.statistics_manager, eval_stats, iteration_log,
                save_path=cfg.run.plot_path, optimal_reward=optimal,
                num_rooms=num_rooms, title=plot_title,
            )

    # Final plot
    eval_stats = evaluator.statistics_manager if use_gating else None
    plot_training_metrics(
        trainer.statistics_manager, eval_stats, iteration_log,
        save_path=cfg.run.plot_path, optimal_reward=optimal,
        num_rooms=num_rooms, title=plot_title,
    )
    rename_plot_with_stats(cfg, iteration_log)

    # Final summary
    sep = "=" * 80
    print(f"\n{sep}")
    print("  Training Complete")
    print(sep)
    print(f"  Best solve rate:    {best_solve_rate:.2f}"
          f" (iteration {best_solve_iter})")
    if iteration_log:
        final_reward = iteration_log[-1]["avg_eval_reward"]
        print(f"  Final avg reward:   {final_reward:+.3f}")
    print(f"  Iterations run:     {len(iteration_log)}")
    print(f"  Accepted:           {n_accepted}/{len(iteration_log)}")
    print()

    # Show final learnt policy
    print("  Final learnt policy:")
    _, total_reward, solved, trace, final_obs = play_greedy_episode(agent)
    action_seq = " -> ".join(name for _, _, name, _, _ in trace)
    status = "SOLVED" if solved else "NOT SOLVED"
    print(f"    {action_seq} -> {status}")
    print()

    # Generate animated GIF of the final policy
    gif_path = exp_dir / "policy_animation.gif"
    render_policy_gif(trace, final_obs, game.env, gif_path, fps=2)
    print(sep)

    # Write eval_summary.csv and eval_manifest.json if ablation is active
    if eval_ablation:
        import csv
        csv_path = exp_dir / "eval_summary.csv"
        fieldnames = ["iteration", "seed", "eval_ablation",
                       "solve_rate", "avg_reward",
                       "abl_solve_rate", "abl_avg_reward", "wall_clock_s"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                     extrasaction="ignore")
            writer.writeheader()
            seed_val = cfg.agent.random_seeds.get("mcts", 0)
            for e in iteration_log:
                row = {
                    "iteration": e["iteration"],
                    "seed": seed_val,
                    "eval_ablation": eval_ablation,
                    "solve_rate": e["solve_rate"],
                    "avg_reward": e["avg_eval_reward"],
                    "abl_solve_rate": e.get("abl_solve_rate", ""),
                    "abl_avg_reward": e.get("abl_avg_reward", ""),
                    "wall_clock_s": e["wall_clock_s"],
                }
                writer.writerow(row)

        import subprocess
        git_hash = ""
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        manifest = {
            "eval_ablation": eval_ablation,
            "git_hash": git_hash,
            "config": cfg._to_serializable_dict() if hasattr(cfg, "_to_serializable_dict") else str(cfg),
        }
        with open(exp_dir / "eval_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    return iteration_log


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Doors Direct Play -- AlphaZero Training")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds for multi-seed runs")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip interactive config editing")
    parser.add_argument("--eval-ablation", type=str, default=None,
                        choices=ABLATION_MODES,
                        help="Eval-only ablation mode (training unchanged)")
    return parser.parse_args()


def _parse_seeds(seeds_str):
    """Parse comma-separated seed string into list of ints."""
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Seed state for interactive editing
    default_seed = str(args.seeds[0]) if args.seeds else "43"
    seed_state = {"seeds_str": default_seed}

    cfg = DoorsDirectConfig()

    # Interactive config editing
    if not args.non_interactive:
        interactive_edit("Doors Direct Play Config",
                         lambda: _build_sections(cfg, seed_state))

    # Determine seeds
    if args.seeds and len(args.seeds) > 1:
        seeds = args.seeds
    else:
        seeds = _parse_seeds(seed_state["seeds_str"])

    # Setup experiment directory
    eval_ablation = args.eval_ablation
    exp_dir = setup_experiment_dir(cfg, eval_ablation=eval_ablation)

    # Build a temporary game for banner display (needs env for layout info)
    from alphazeropp.instances.doors.game import DoorsDirectGame
    _banner_game = DoorsDirectGame(**cfg.game.kwargs)

    # Print startup info
    print_banner(cfg, _banner_game, exp_dir)
    print_architecture(cfg)

    # Save config
    cfg.save(str(exp_dir / "config.json"))

    optimal = _optimal_reward(
        cfg.game.kwargs.get("num_rooms", 2),
        cfg.game.kwargs.get("step_penalty", 0.01),
        cfg.game.kwargs.get("unlock_bonus", 0.1))

    if len(seeds) <= 1:
        # Single-seed run
        if seeds:
            s = seeds[0]
            cfg.agent.random_seeds = {
                "mcts": s, "train": s + 1,
                "eval": s + 2, "external_policy": s + 3,
            }
        _run_single_seed(cfg, exp_dir, eval_ablation=eval_ablation)
    else:
        # Multi-seed run
        seed_dirs = {}
        for seed in seeds:
            print()
            print(f"{'='*70}")
            print(f"  SEED {seed}")
            print(f"{'='*70}")
            cfg.agent.random_seeds = {
                "mcts": seed, "train": seed + 1,
                "eval": seed + 2, "external_policy": seed + 3,
            }
            seed_dir = exp_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            cfg.trainer.checkpoint_dir = str(seed_dir / "checkpoints")
            cfg.run.plot_path = str(seed_dir / "training_metrics.png")
            seed_dirs[seed] = seed_dir
            _run_single_seed(cfg, seed_dir, eval_ablation=eval_ablation)

        # Aggregate across seeds
        logs = _load_iteration_logs(seed_dirs)
        train_stats = _load_train_stats(seed_dirs)
        _print_seed_summary(logs, seeds)
        _plot_seed_comparison(logs, seeds, exp_dir,
                              optimal_reward=optimal,
                              train_stats=train_stats)

        print()
        print("[Analysis] Plots saved. Key things to look for:")
        print("  - seed_comparison.png: Do seeds converge? "
              "High variance = unstable training")


if __name__ == "__main__":
    main()
