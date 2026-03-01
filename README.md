
# Chef’s Hat Gym – Reinforcement Learning Agent

Student ID: 16365513
Assigned Variant: Sparse / Delayed Reward Variant (ID mod 7 = 3)

# Task Overview

This project implements, modifies, and evaluates reinforcement learning agents in the official Chef’s Hat Gym multi-agent card game environment.

For this assignment, the original Chef’s Hat DQN implementation was extended and improved, and a new experimental script was created to enable controlled comparison between sparse and shaped reward strategies.

The objective of this project is to:

Design and train RL agents in a delayed reward setting
Investigate the impact of sparse terminal rewards
Implement reward shaping for improved credit assignment
Compare RAW vs SHAPED reward strategies
Evaluate learning stability and performance

Two agents are implemented and compared:

RAW reward agent (terminal reward only)
SHAPED reward agent (step-level shaping + terminal reward)

# Modifications and New Files

This project includes significant modifications to the original implementation:

### Modified File

`src/agents/agent_dqn.py`

Enhancements include:

 Dueling Double DQN architecture
 Batch Normalisation for stability
 Soft target network updates (τ = 0.01)
 Gradient clipping
 Improved state encoding (40-dimensional representation)
 Explicit action masking
 Custom reward shaping mechanism
 Replay multiple times per match
 Structured tracking of training statistics



### Newly Created File

`src/reward_comparison.py`

This file was created to:

 Train RAW and SHAPED agents separately
 Evaluate both agents over controlled test matches
 Generate cumulative win rate plots
 Generate training comparison plots
 Print structured experimental summaries
 Automatically compute win-rate improvement

This file is the main entry point for experimentation.


# Chef’s Hat Gym Environment

Official environment used:

 GitHub: [https://github.com/pablovin/ChefsHatGYM](https://github.com/pablovin/ChefsHatGYM)
 Documentation: [https://chefshatgym.readthedocs.io/en/latest/](https://chefshatgym.readthedocs.io/en/latest/)

Key characteristics:

 4-player multi-agent competition
 Large discrete action space (~200 actions)
 Delayed terminal rewards
 Non-stationary environment
 OpenAI Gym–compatible API



# Installation Instructions

## Step 1: Clone Repository


git clone https://github.com/aliinayath3-hash/7043scn
cd REPOSITORY


## Step 2: Install Dependencies


pip install tensorflow numpy matplotlib gym asyncio


---

# How to Run Experiments

Navigate to the src directory:


cd src


Run:


python reward_comparison.py


This script will:

 Train RAW reward agent (1500 matches)
 Train SHAPED reward agent (1500 matches)
 Evaluate both agents (200 matches)
 Generate comparison plots
 Print win-rate comparison summary

---

# Experimental Setup

Training matches: 1500
Testing matches: 200
Opponents: 3 Random baseline agents

Algorithm:

 Dueling Double DQN
 Experience Replay
 Double Q-learning target selection
 Soft Target Updates (τ = 0.01)
 Batch Normalisation
 Huber Loss
 Gradient Clipping
 ε-greedy exploration

Final epsilon after training: ≈ 0.05

---

# State Representation (40 Dimensions)

The agent uses a structured encoding consisting of:

 13 dims: Hand rank histogram
 13 dims: Board rank histogram
 4 dims: Player hand sizes (normalised)
 4 dims: Player positions
 8 dims: Game statistics (hand size, board size, pass rate, step fraction, valid actions, etc.)

This representation captures:

 Local card information
 Global competitive context
 Action feasibility

This improves learning in a delayed, multi-agent setting.

---

# Reward Strategy (Sparse / Delayed Variant)

## RAW Agent

Uses only terminal reward:

 1st place: +10
 2nd place: +2
 3rd place: −1
 4th place: −3

This represents a purely delayed reward setting.

---

## SHAPED Agent

Adds intermediate shaping rewards:

 +0.05 × cards played
 −0.05 for passing
 −0.005 time penalty per step
 Terminal reward remains unchanged

Purpose:

 Improve credit assignment
 Accelerate early learning
 Reduce sparse reward difficulty

---

# Evaluation Metrics

Agents are evaluated using:

 Win Rate (primary metric)
 Cumulative win-rate curve
 Training position moving average
 Reward curves
 Training loss curves

Random baseline win rate ≈ 0.25 (4-player game)

---

# Final Experimental Results

Test matches: 200

RAW Agent:

 Win rate: 0.040 (8/200 wins)

SHAPED Agent:

 Win rate: 0.075 (15/200 wins)

Improvement:

 Δ (SHAPED − RAW): +0.035

Although both agents remain below the random baseline (0.25), reward shaping improved performance relative to purely sparse terminal rewards.

This demonstrates that intermediate feedback partially mitigates delayed reward challenges, even in a highly non-stationary multi-agent environment.

---

# Generated Outputs

RAW Training:


outputs_raw/
    rewards.png
    rewards_smoothed.png
    positions.png
    loss.png


RAW Test:


outputs_raw_test/
    winrate_curve.png


SHAPED Training:


outputs_shaped/
    rewards.png
    rewards_smoothed.png
    positions.png
    loss.png


SHAPED Test:


outputs_shaped_test/
    winrate_curve.png


Comparison:

comparison_winrate.png
comparison_training_positions.png




# Limitations

 Performance remains below random baseline
 Multi-agent non-stationarity increases instability
 Large action space complicates exploration
 Random opponents limit generalisation analysis
 Training time significant on CPU

---

# Reproducibility

To reproduce:

cd src
python reward_comparison.py

No external datasets required.

All experiments are fully contained within the repository.



# Repository Structure
root/
│
├── src/
│   ├── agents/
│   │   ├── agent_dqn.py        (modified)
│   │   ├── random_agent.py
│   │
│   ├── reward_comparison.py    (new file)
│   ├── rooms/
│
├── outputs_raw/
├── outputs_shaped/
├── outputs_raw_test/
├── outputs_shaped_test/
│
└── README.md



# Conclusion

This project demonstrates:

Correct integration of Chef’s Hat Gym
Implementation of an improved Dueling Double DQN
Handling of delayed and sparse reward conditions
Controlled experimental comparison
Structured evaluation and visualisation

Reward shaping provided measurable improvement over purely terminal rewards, confirming the importance of intermediate feedback in sparse reward reinforcement learning environments.

