"""
Generate a reward reference PNG for the Doors environment.

Produces two tables:
  1. In-game reward events (per-step rewards)
  2. Possible program outcomes with cumulative reward ranges

Usage:
    python scripts/generate_reward_reference.py                # D=3 (default)
    python scripts/generate_reward_reference.py --num-rooms 5  # D=5
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_reward_reference(num_rooms, step_penalty=0.01, unlock_bonus=0.1,
                               locs_per_room=2, save_dir="experiments"):
    D = num_rooms
    K = D - 1  # number of keys
    horizon = 5 * (2 * K + 1)
    optimal_steps = 2 * K + 1

    # --- Table 1: In-game reward events ---
    events_header = ["Event", "Reward"]
    events_data = [
        ["Each step taken", f"{-step_penalty:+.2f}"],
        ["Pick up key (unlocks new room)", f"+{unlock_bonus:.2f}"],
        ["Pick up key (room already open)", "+0.00"],
        ["Reach goal location", "+1.00"],
        ["Horizon timeout (episode ends)", "+0.00"],
        ["NOOP / invalid action", "+0.00"],
    ]

    # --- Table 2: Program outcome categories ---
    outcomes_header = ["Outcome", "Keys", "Steps", "Reward", "Note"]
    outcomes_data = []

    # Timeout outcomes (no goal reached)
    for keys in range(K + 1):
        bonus = keys * unlock_bonus
        # Best case: minimum steps to pick those keys
        min_steps = max(2 * keys + 1, 1) if keys > 0 else 1
        best_reward = bonus - min_steps * step_penalty
        worst_reward = bonus - horizon * step_penalty
        if keys == 0:
            note = "Random walk, never picks keys"
        elif keys == K:
            note = "All keys but never reaches goal"
        else:
            note = f"Picks {keys}/{K} keys, stuck"
        outcomes_data.append([
            f"Timeout ({keys} keys)",
            str(keys),
            f"{min_steps}–{horizon}",
            f"{worst_reward:+.2f} to {best_reward:+.2f}",
            note,
        ])

    # Goal reached outcomes
    for keys in range(K + 1):
        bonus = keys * unlock_bonus
        # Need to traverse at least some rooms to reach goal
        if keys < K:
            # Can't actually reach goal without all keys in standard layout
            # but partial-key goal reach is possible if goal is in room 0
            # For standard layout, goal is in last room, needs all keys
            # Still show for completeness
            min_steps_goal = 2 * keys + 1
            note = f"Reaches goal early (only room 0 reachable)" if keys == 0 else "Partial keys"
        else:
            min_steps_goal = optimal_steps
            note = "OPTIMAL"
        best_reward = 1.0 + bonus - min_steps_goal * step_penalty
        worst_reward = 1.0 + bonus - horizon * step_penalty
        outcomes_data.append([
            f"Goal ({keys} keys)",
            str(keys),
            f"{min_steps_goal}–{horizon}",
            f"{worst_reward:+.2f} to {best_reward:+.2f}",
            note,
        ])

    optimal_reward = 1.0 + K * unlock_bonus - optimal_steps * step_penalty

    # --- Render ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6 + 0.4 * len(outcomes_data)))
    fig.suptitle(f"Doors Reward Reference — D={D} rooms, {K} keys\n"
                 f"step_penalty={step_penalty}, unlock_bonus={unlock_bonus}, "
                 f"horizon={horizon}, optimal={optimal_reward:+.2f}",
                 fontsize=12, fontweight="bold")

    # Table 1: Events
    ax1.axis("off")
    ax1.set_title("Per-Step Reward Events", fontsize=11, fontweight="bold",
                  loc="left", pad=10)
    table1 = ax1.table(
        cellText=events_data,
        colLabels=events_header,
        loc="center",
        cellLoc="left",
        colWidths=[0.5, 0.15],
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.4)
    for (row, col), cell in table1.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D6E4F0")

    # Table 2: Outcomes
    ax2.axis("off")
    ax2.set_title("Possible Program Outcomes (Cumulative Reward)",
                  fontsize=11, fontweight="bold", loc="left", pad=10)
    table2 = ax2.table(
        cellText=outcomes_data,
        colLabels=outcomes_header,
        loc="center",
        cellLoc="left",
        colWidths=[0.22, 0.06, 0.12, 0.22, 0.38],
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.4)
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif "OPTIMAL" in (outcomes_data[row - 1][-1] if row > 0 and row <= len(outcomes_data) else ""):
            cell.set_facecolor("#C6EFCE")
        elif row <= K + 1:  # timeout rows
            cell.set_facecolor("#FFF2CC") if row % 2 == 1 else cell.set_facecolor("#FFE699")
        else:
            cell.set_facecolor("#D6E4F0") if row % 2 == 0 else cell.set_facecolor("#E9F0F8")

    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"reward_reference_D{D}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward reference to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Doors reward reference PNG")
    parser.add_argument("--num-rooms", type=int, default=3,
                        help="Number of rooms D (default: 3)")
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--unlock-bonus", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    generate_reward_reference(
        args.num_rooms,
        step_penalty=args.step_penalty,
        unlock_bonus=args.unlock_bonus,
        save_dir=args.output_dir,
    )
