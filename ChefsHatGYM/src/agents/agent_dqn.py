import os
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Lambda, Add, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.losses import Huber
from .base_agent import BaseAgent


def dueling_lambda(a):
    return a - tf.reduce_mean(a, axis=1, keepdims=True)


class DQNAgent(BaseAgent):

    def __init__(
        self,
        name="",
        state_size=40,          
        action_size=200,
        gamma=0.97,             
        lr=3e-4,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,    
        batch_size=128,         
        memory_size=50000,      
        log_directory="",
        verbose_console=False,
        train=True,
        model_path=None,
        load_model=False,
        reward_mode="shaped",
        run_remote=False,
        host="localhost",
        port=8765,
        room_name="room",
        room_password="password",
        replay_per_match=5,     
    ):
        super().__init__(
            name,
            log_directory,
            verbose_console,
            run_remote,
            host,
            port,
            room_name,
            room_password,
        )

        self.reward_mode = reward_mode
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.epsilon = epsilon if train else 0.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train = train
        self.model_path = model_path
        self.replay_per_match = replay_per_match

        self.last_state = None
        self.last_action = None
        self.last_mask = None

        # Track hand size for shaping
        self.last_hand_size = None
        self.hand_size = None
        self.match_step = 0

        self.positions = []
        self.rewards = []
        self.loss_history = []
        self.epsilon_history = []

        self.all_actions = None

        # Target update counter
        self._update_counter = 0
        self._target_update_freq = 10  # hard update every N replays

        model_path = os.path.join(log_directory, "model", "dql_model.h5")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if load_model and os.path.exists(model_path):
            self.model = keras_load_model(
                model_path, custom_objects={"dueling_lambda": dueling_lambda}
            )
            target_path = model_path.replace(".h5", ".target.h5")
            if os.path.exists(target_path):
                self.target_model = keras_load_model(
                    target_path,
                    custom_objects={"dueling_lambda": dueling_lambda},
                )
            else:
                self.target_model = self._build_model(lr)
                self.target_model.set_weights(self.model.get_weights())
        else:
            self.model = self._build_model(lr)
            self.target_model = self._build_model(lr)
            self.target_model.set_weights(self.model.get_weights())

        if not self.train:
            self.epsilon = 0.0

    # ===========================
    # MODEL
    # ===========================

    def _build_model(self, lr):
        """Dueling DQN with batch normalisation for stability."""
        state_input = Input(shape=(self.state_size,))
        x = Dense(512, activation="relu")(state_input)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation="relu")(x)

        value = Dense(64, activation="relu")(x)
        value = Dense(1)(value)

        advantage = Dense(64, activation="relu")(x)
        advantage = Dense(self.action_size)(advantage)
        advantage_mean = Lambda(dueling_lambda)(advantage)

        q_values = Add()([value, advantage_mean])

        model = Model(inputs=state_input, outputs=q_values)
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=lr, clipnorm=1.0))
        return model

    # ===========================
    # STATE ENCODING
    # ===========================

    def _encode_state(self, observations):
        """
        Richer 40-dim state vector:
          - 13 dims: normalised hand card counts (rank 1-13)
          - 13 dims: normalised board top card counts
          - 4  dims: number of cards each player has (normalised), self last
          - 4  dims: player order / positions normalised
          - 4  dims: hand size, board size, pass fraction, step fraction
          - 2  dims: # valid actions (norm), can_pass flag
        """
        hand = np.array(observations["hand"]).flatten()
        board = np.array(observations["board"]).flatten()

        # Hand histogram (rank counts, normalised by 4)
        hand_hist = np.zeros(13, dtype=np.float32)
        for card in hand:
            idx = int(card) % 13
            hand_hist[idx] += 1
        hand_hist /= 4.0

        # Board top histogram
        board_hist = np.zeros(13, dtype=np.float32)
        for card in board:
            idx = int(card) % 13
            board_hist[idx] += 1
        board_hist /= 4.0

        # Player card counts (4 players), normalised by 17
        player_counts = np.array(
            observations.get("players_hand_size", [17, 17, 17, len(hand)]),
            dtype=np.float32
        ) / 17.0
        if len(player_counts) < 4:
            player_counts = np.pad(player_counts, (0, 4 - len(player_counts)), constant_values=0)
        player_counts = player_counts[:4]

        # Positions
        player_positions = np.array(
            observations.get("players_position", [0, 0, 0, 0]),
            dtype=np.float32
        ) / 4.0
        if len(player_positions) < 4:
            player_positions = np.pad(player_positions, (0, 4 - len(player_positions)), constant_values=0)
        player_positions = player_positions[:4]

        # Scalars
        possible_vals = list(observations["possible_actions"])
        n_valid = len(possible_vals)
        can_pass = 1.0 if "pass" in [str(a).lower() for a in possible_vals] else 0.0
        hand_size = len(hand) / 17.0
        board_size = len(board) / 17.0
        pass_rate = getattr(self, "_pass_count", 0) / max(self.match_step, 1)
        step_frac = min(self.match_step / 50.0, 1.0)

        scalars = np.array([
            hand_size, board_size,
            min(n_valid / 50.0, 1.0), can_pass,
            pass_rate, step_frac, 0.0, 0.0  # 8 dims total (padded)
        ], dtype=np.float32)

        state = np.concatenate([hand_hist, board_hist, player_counts, player_positions, scalars])
        # Ensure correct size
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))
        return state[:self.state_size].astype(np.float32)

    # ===========================
    # MEMORY + TRAINING
    # ===========================

    def remember(self, *transition):
        if self.train:
            self.memory.append(transition)

    def act(self, state, mask, valid_indices):
        if self.train and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_indices))

        # Fast inference: use __call__ instead of predict()
        q_values = self.model(state[np.newaxis, :], training=False).numpy()[0]
        masked_q = np.where(mask, q_values, -np.inf)
        return int(np.argmax(masked_q))

    def replay(self):
        if not self.train or len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states      = np.array([b[0] for b in batch], dtype=np.float32)
        masks_next  = np.array([b[5] for b in batch], dtype=np.float32)
        actions     = np.array([b[2] for b in batch])
        rewards     = np.array([b[3] for b in batch], dtype=np.float32)
        next_states = np.array([b[4] for b in batch], dtype=np.float32)
        dones       = np.array([b[6] for b in batch])

        # Double DQN target
        next_q_online = self.model(next_states, training=False).numpy()
        next_q_target = self.target_model(next_states, training=False).numpy()

        target = self.model(states, training=False).numpy()

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # Double DQN: select action with online, evaluate with target
                valid_next = np.where(masks_next[i], next_q_online[i], -np.inf)
                best_next_action = int(np.argmax(valid_next))
                target[i][actions[i]] = (
                    rewards[i] + self.gamma * next_q_target[i][best_next_action]
                )

        history = self.model.fit(states, target, epochs=1, verbose=0, batch_size=self.batch_size)

        self.loss_history.append(float(history.history["loss"][0]))
        self.epsilon_history.append(self.epsilon)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Soft update
        self._update_counter += 1
        tau = 0.01  # slower soft update for stability
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = [
            tau * mw + (1 - tau) * tw
            for mw, tw in zip(model_weights, target_weights)
        ]
        self.target_model.set_weights(new_weights)

    # ===========================
    # REWARD SHAPING (FIXED)
    # ===========================

    def _shaped_reward(self, action_str, prev_hand_size, curr_hand_size):
        
        reward = 0.0
        cards_played = (prev_hand_size or 0) - (curr_hand_size or 0)
        if cards_played > 0:
            reward += 0.05 * cards_played   # reward for emptying hand faster

        # Only penalise passing if there were real alternatives
        if action_str.lower() == "pass":
            reward -= 0.05

        reward -= 0.005   # small time penalty per step
        return reward

    # ===========================
    # GAME LIFECYCLE
    # ===========================

    def update_game_start(self, info):
        if "actions" in info:
            self.all_actions = list(info["actions"].values())

    def update_new_hand(self, payload):
        self.last_state = None
        self.last_action = None
        self.last_mask = None
        self.last_hand_size = None
        self.match_step = 0
        self._pass_count = 0

    def request_cards_to_exchange(self, payload):
        return sorted(payload["hand"])[-payload["n"]:]

    def request_special_action(self, payload):
        return True

    def request_action(self, observations):
        self.match_step += 1
        hand = np.array(observations["hand"]).flatten()
        curr_hand_size = len(hand)

        state = self._encode_state(observations)

        possible_vals = list(observations["possible_actions"])
        mask = np.zeros(self.action_size, dtype=np.float32)
        valid_indices = [self.all_actions.index(v) for v in possible_vals]
        mask[valid_indices] = 1.0

        action = self.act(state, mask, valid_indices)
        action_str = self.all_actions[action]

        if str(action_str).lower() == "pass":
            self._pass_count = getattr(self, "_pass_count", 0) + 1

        # Compute intermediate shaping reward
        if self.reward_mode == "shaped":
            step_reward = self._shaped_reward(
                str(action_str), self.last_hand_size, curr_hand_size
            )
        else:
            step_reward = 0.0

        if self.last_state is not None and self.train:
            self.remember(
                self.last_state,
                self.last_mask,
                self.last_action,
                step_reward,
                state,
                mask,
                False,
            )

        self.last_state = state
        self.last_action = action
        self.last_mask = mask
        self.last_hand_size = curr_hand_size

        return action

    def update_match_over(self, payload):
        finishing_order = payload.get("finishing_order", [])

        try:
            place = finishing_order.index(self.name) + 1
        except ValueError:
            place = 4

        # Reward gradient calibrated to Chef's Hat 4-player game
        place_rewards = {1: 10.0, 2: 2.0, 3: -1.0, 4: -3.0}
        reward = place_rewards.get(place, -3.0)

        self.positions.append(place)
        self.rewards.append(reward)

        if self.last_state is not None and self.train:
            self.remember(
                self.last_state,
                self.last_mask,
                self.last_action,
                reward,
                self.last_state,
                self.last_mask,
                True,
            )

        # Replay multiple times at end of each match
        for _ in range(self.replay_per_match):
            self.replay()

    def save_model(self):
        if self.model_path:
            self.model.save(self.model_path)
            self.target_model.save(self.model_path.replace(".h5", ".target.h5"))

    # ===========================
    # PLOTTING FUNCTIONS
    # ===========================

    def plot_rewards(self, path_raw, path_avg, window=30):
        import matplotlib.pyplot as plt

        if len(self.rewards) == 0:
            return

        rewards = np.array(self.rewards)
        x = np.arange(len(rewards))

        plt.figure()
        plt.plot(x, rewards, alpha=0.6)
        plt.title("Reward per Match")
        plt.xlabel("Match")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(path_raw)
        plt.close()

        window = min(window, len(rewards))
        avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x_avg = np.arange(window - 1, len(rewards))

        plt.figure()
        plt.plot(x, rewards, alpha=0.2, label="Raw")
        plt.plot(x_avg, avg, label=f"MA-{window}")
        plt.title("Smoothed Reward")
        plt.xlabel("Match")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_avg)
        plt.close()

    def plot_positions(self, path):
        import matplotlib.pyplot as plt

        if len(self.positions) == 0:
            return

        positions = np.array(self.positions)
        window = min(50, len(positions))
        avg = np.convolve(positions, np.ones(window) / window, mode="valid")
        x_avg = np.arange(window - 1, len(positions))

        plt.figure()
        plt.plot(positions, alpha=0.3, label="Raw")
        plt.plot(x_avg, avg, label=f"MA-{window}")
        plt.gca().invert_yaxis()
        plt.title("Position per Match (lower=better)")
        plt.xlabel("Match")
        plt.ylabel("Position (1=Best)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_loss(self, path):
        import matplotlib.pyplot as plt

        if len(self.loss_history) == 0:
            return

        loss = np.array(self.loss_history)
        window = min(50, len(loss))
        avg = np.convolve(loss, np.ones(window) / window, mode="valid")

        plt.figure()
        plt.plot(loss, alpha=0.3, label="Loss")
        plt.plot(np.arange(window - 1, len(loss)), avg, label=f"MA-{window}")
        plt.title("Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()