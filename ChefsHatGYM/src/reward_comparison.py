import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt

from agents.random_agent import RandomAgent
from agents.agent_dqn import DQNAgent
from rooms.room import Room


# ---- matches count ---------------------------------------------------------------------
TRAIN_MATCHES = 1500   
TEST_MATCHES  = 200
# -------------------------------------------------------------------------------------------


def run_experiment(reward_mode, train_matches, test_matches):

    print(f"\n{'='*55}")
    print(f"  Experiment: {reward_mode.upper()}  "
          f"(train={train_matches}, test={test_matches})")
    print(f"{'='*55}")

    # --- TRAINING ---
    train_room = Room(
        run_remote_room=False,
        room_name=f"Room_{reward_mode}",
        max_matches=train_matches,
        output_folder=f"outputs_{reward_mode}",
        save_game_dataset=False,   
        save_logs_game=False,
        save_logs_room=False,
    )

    for i in range(3):
        train_room.connect_player(
            RandomAgent(
                name=f"Random{i}",
                log_directory=train_room.room_dir,
                verbose_log=False,
            )
        )

    model_path = os.path.join(train_room.room_dir, "model", "dql_model.h5")

    dqn_agent = DQNAgent(
        name="DQL",
        train=True,
        log_directory=train_room.room_dir,
        verbose_console=False,
        model_path=model_path,
        load_model=False,
        reward_mode=reward_mode,
    )

    train_room.connect_player(dqn_agent)
    asyncio.run(train_room.run())

    # Save model after training
    dqn_agent.save_model()

    # Progress report every 100 matches during training
    _print_training_summary(dqn_agent, reward_mode, train_matches)

    # --- TRAINING PLOTS ---
    dqn_agent.plot_rewards(
        os.path.join(train_room.room_dir, "rewards.png"),
        os.path.join(train_room.room_dir, "rewards_smoothed.png"),
    )
    dqn_agent.plot_positions(
        os.path.join(train_room.room_dir, "positions.png")
    )
    dqn_agent.plot_loss(
        os.path.join(train_room.room_dir, "loss.png")
    )

    # --- TESTING ---
    test_room = Room(
        run_remote_room=False,
        room_name=f"Room_{reward_mode}_TEST",
        max_matches=test_matches,
        output_folder=f"outputs_{reward_mode}_test",
        save_game_dataset=False,
        save_logs_game=False,
        save_logs_room=False,
    )

    for i in range(3):
        test_room.connect_player(
            RandomAgent(
                name=f"Random{i}",
                log_directory=test_room.room_dir,
                verbose_log=False,
            )
        )

    test_agent = DQNAgent(
        name="DQL",
        train=False,
        log_directory=test_room.room_dir,
        verbose_console=False,
        model_path=model_path,
        load_model=True,
        reward_mode=reward_mode,
    )

    test_room.connect_player(test_agent)
    asyncio.run(test_room.run())

    # --- Win-rate ---
    wins     = sum(1 for p in test_agent.positions if p == 1)
    win_rate = wins / max(len(test_agent.positions), 1)

    print(f"\n  {reward_mode.upper()} → win rate: {win_rate:.3f}  "
          f"({wins}/{len(test_agent.positions)} wins)")

    # --- Win-rate curve ---
    cumulative_wr = (
        np.cumsum([1 if p == 1 else 0 for p in test_agent.positions])
        / np.arange(1, len(test_agent.positions) + 1)
    )

    plt.figure()
    plt.axhline(0.25, color="grey", linestyle="--", label="Random baseline (0.25)")
    plt.plot(cumulative_wr, label=f"{reward_mode.upper()} agent")
    plt.title(f"{reward_mode.upper()} – Cumulative Win Rate")
    plt.xlabel("Match")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(test_room.room_dir, "winrate_curve.png"))
    plt.close()

    return win_rate, test_agent.positions


def _print_training_summary(agent, mode, total):
    """Print a concise per-100-match breakdown."""
    positions = agent.positions
    chunk = 100
    print(f"\n  {mode.upper()} Training summary (win rate per {chunk} matches):")
    for start in range(0, len(positions), chunk):
        chunk_pos = positions[start:start + chunk]
        wr = sum(1 for p in chunk_pos if p == 1) / len(chunk_pos)
        print(f"    Match {start+1:4d}-{start+len(chunk_pos):4d}  WR={wr:.3f}")
    final_eps = agent.epsilon if agent.train else 0.0
    print(f"  Final epsilon: {final_eps:.4f}")


# ----- COMPARISON PLOT ----

def plot_comparison(raw_positions, shaped_positions):

    def cumwr(positions):
        return (
            np.cumsum([1 if p == 1 else 0 for p in positions])
            / np.arange(1, len(positions) + 1)
        )

    raw_wr    = cumwr(raw_positions)
    shaped_wr = cumwr(shaped_positions)

    plt.figure(figsize=(9, 5))
    plt.axhline(0.25, color="grey", linestyle="--", label="Random baseline")
    plt.plot(raw_wr,    label="RAW reward",    linewidth=2)
    plt.plot(shaped_wr, label="SHAPED reward", linewidth=2)
    plt.xlabel("Test Match")
    plt.ylabel("Cumulative Win Rate")
    plt.title("Reward Shaping Comparison – Test Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_winrate.png")
    plt.close()
    print("\n  Comparison plot saved as comparison_winrate.png")


def plot_training_curves(raw_agent_positions, shaped_agent_positions,
                         raw_dir, shaped_dir):
    """Side-by-side training position curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, positions, label in zip(
        axes,
        [raw_agent_positions, shaped_agent_positions],
        ["RAW", "SHAPED"],
    ):
        arr = np.array(positions)
        window = min(50, len(arr))
        avg = np.convolve(arr, np.ones(window) / window, mode="valid")
        ax.plot(arr, alpha=0.25)
        ax.plot(np.arange(window - 1, len(arr)), avg, linewidth=2)
        ax.axhline(2.5, color="grey", linestyle="--", label="Random avg")
        ax.invert_yaxis()
        ax.set_title(f"{label} – Training Positions")
        ax.set_xlabel("Match")
        ax.set_ylabel("Position (1=Best)")
        ax.legend()
    plt.tight_layout()
    plt.savefig("comparison_training_positions.png")
    plt.close()


# ------ MAIN------

if __name__ == "__main__":

    raw_wr, raw_positions = run_experiment("raw", TRAIN_MATCHES, TEST_MATCHES)

    shaped_wr, shaped_positions = run_experiment("shaped", TRAIN_MATCHES, TEST_MATCHES)

    plot_comparison(raw_positions, shaped_positions)

    print("\n" + "=" * 55)
    print("  FINAL RESULTS")
    print("=" * 55)
    print(f"  RAW    win rate : {raw_wr:.3f}  (random baseline ≈ 0.250)")
    print(f"  SHAPED win rate : {shaped_wr:.3f}  (random baseline ≈ 0.250)")
    delta = shaped_wr - raw_wr
    print(f"   SHAPED – RAW  : {delta:+.3f}")
    print("=" * 55)