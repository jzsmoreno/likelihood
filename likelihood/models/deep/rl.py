import random
from collections import deque

import numpy as np
import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) > version.parse("2.15.0"):
    from ._autoencoders import AutoClassifier
else:
    from .autoencoders import AutoClassifier


def print_progress_bar(iteration, total, length=30):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\rProgress: |{bar}| {percent}% Complete", end="\r")
    if iteration == total:
        print()


class Env:
    def __init__(self, model, maxlen=100, name="likenasium"):
        """
        Initialize the environment with a model.

        Parameters
        ----------
        model : Any
            Model with `.predict()` method (e.g., Keras model).
        maxlen : int
            Maximum length of deque. By default it is set to `100`.
        name : str
            The name of the environment. By default it is set to `likenasium`.
        """
        self.model = model
        self.maxlen = maxlen
        self.transitions = deque(
            maxlen=self.maxlen
        )  # Stores (state, action, reward, next_action, done)
        self.current_state = None
        self.current_step = 0
        self.done = False

    def step(self, state, action, verbose=0):
        """
        Perform an environment step with the given action.

        Parameters
        ----------
        state : `np.ndarray`
            Current state to process (input to the model).
        action : `int`
            Expected action to process.

        Returns
        -------
            `tuple` : (current_state, action_pred, reward, next_action, done)
        """
        if self.done:
            return None, None, 0, None, True

        # Process action through model
        model_output = self.model.predict(state.reshape((1, -1)), verbose=verbose)
        action_pred = np.argmax(model_output, axis=1)[0]
        model_output[:, action_pred] = 0.0
        next_action = np.max(model_output, axis=1)[0]  # Second most probable action

        # Calculate reward (1 if correct prediction, 0 otherwise)
        reward = 1 if action_pred == action else 0

        # Update current state
        self.current_state = state
        self.current_step += 1

        # Add transition to history
        if self.current_step <= self.maxlen:
            self.transitions.append(
                (
                    self.current_state,  # Previous state
                    action_pred,  # Current action
                    reward,  # Reward
                    next_action,  # Next action
                    self.done,  # Done flag
                )
            )
        return self.current_state, action_pred, reward, next_action, self.done

    def reset(self):
        """Reset the environment to initial state."""
        self.current_state = None
        self.current_step = 0
        self.done = False
        self.transitions = deque(maxlen=self.maxlen)
        return self.current_state

    def get_transitions(self):
        """Get all stored transitions."""
        return self.transitions


class AutoQL:
    """
    AutoQL: A reinforcement learning agent using Q-learning with Epsilon-greedy policy.

    This class implements a Q-learning agent with:
    - Epsilon-greedy policy for exploration
    - Replay buffer for experience replay
    - Automatic model version handling for TensorFlow
    """

    def __init__(
        self,
        env,
        model,
        maxlen=2000,
    ):
        """Initialize AutoQL agent

        Parameters
        ----------
        env : `Any`
            The environment to interact with
        model : `tf.keras.Model`
            The Q-network model
        """

        self.env = env
        self.model = model
        self.maxlen = maxlen
        self.replay_buffer = deque(maxlen=self.maxlen)

    def epsilon_greedy_policy(self, state, action, epsilon=0):
        """
        Epsilon-greedy policy for action selection

        Parameters
        ----------
        state : `np.ndarray`
            Current state.
        action : `int`
            Expected action to process.
        epsilon : `float`
            Exploration probability. By default it is set to `0`

        Returns
        -------
            `tuple` : (state, action, reward, next_action, done)
        """
        current_state, value, reward, next_action, done = self.env.step(state, action)

        if np.random.rand() > epsilon:
            state = np.asarray(state).astype(np.float32)
            return current_state, value, reward, next_action, done
        step_ = random.sample(self.env.get_transitions(), 1)
        _state, greedy_action, _reward, _next_action, _done = zip(*step_)

        return _state[0], greedy_action[0], _reward[0], _next_action[0], _done[0]

    def play_one_step(self, state, action, epsilon):
        """
        Perform one step in the environment and add experience to buffer

        Parameters
        ----------
        state : `np.ndarray`
            Current state
        action : `int`
            Expected action to process.

        epsilon : `float`
            Exploration probability.

        Returns
        -------
            `tuple` : (state, action, reward, next_action, done)
        """
        current_state, greedy_action, reward, next_action, done = self.epsilon_greedy_policy(
            state, action, epsilon
        )

        done = 1 if done else 0

        # Add experience to replay buffer
        self.replay_buffer.append(
            (
                current_state,  # Previous state
                greedy_action,  # Current action
                reward,  # Reward
                next_action,  # Next action
                done,  # Done flag
            )
        )

        return current_state, greedy_action, reward, next_action, done

    @tf.function
    def _training_step(self):
        """
        Perform one training step using experience replay

        Returns
        -------
            `float` : Training loss
        """

        batch_ = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_actions, dones = zip(*batch_)
        states = np.array(states).reshape(self.batch_size, -1)
        actions = np.array(actions).reshape(
            self.batch_size,
        )
        rewards = np.array(rewards).reshape(
            self.batch_size,
        )
        max_next_Q_values = np.array(next_actions).reshape(self.batch_size, -1)
        dones = np.array(dones).reshape(
            self.batch_size,
        )
        target_Q_values = rewards + (1 - dones) * self.gamma * max_next_Q_values

        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        target_Q_values = tf.convert_to_tensor(target_Q_values, dtype=tf.float32)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            Q_values = tf.gather_nd(all_Q_values, indices)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(
        self,
        x_data,
        y_data,
        optimizer="adam",
        loss_fn="mse",
        num_episodes=50,
        num_steps=100,
        gamma=0.7,
        batch_size=32,
        patience=10,
        alpha=0.01,
    ):
        """Train the agent for a fixed number of episodes

        Parameters
        ----------
        optimizer : `str`
            The optimizer for training (e.g., `sgd`). By default it is set to `adam`.
        loss_fn : `str`
            The loss function. By default it is set to `mse`.
        num_episodes : `int`
            Total number of episodes to train. By default it is set to `50`.
        num_steps : `int`
            Steps per episode. By default it is set to `100`. If `num_steps` is less than `self.env.maxlen`, then the second will be chosen.
        gamma : `float`
            Discount factor. By default it is set to `0.7`.
        batch_size : `int`
            Size of training batches. By default it is set to `32`.
        patience : `int`
            How many episodes to wait for improvement.
        alpha : `float`
            Trade-off factor between loss and reward.
        """
        rewards = []
        self.best_weights = None
        self.best_loss = float("inf")

        optimizers = {
            "sgd": tf.keras.optimizers.SGD(),
            "adam": tf.keras.optimizers.Adam(),
            "adamw": tf.keras.optimizers.AdamW(),
            "adadelta": tf.keras.optimizers.Adadelta(),
            "rmsprop": tf.keras.optimizers.RMSprop(),
        }
        self.optimizer = optimizers[optimizer]
        losses = {
            "mse": tf.keras.losses.MeanSquaredError(),
            "mae": tf.keras.losses.MeanAbsoluteError(),
            "mape": tf.keras.losses.MeanAbsolutePercentageError(),
        }
        self.loss_fn = losses[loss_fn]
        self.num_episodes = num_episodes
        self.num_steps = num_steps if num_steps >= self.env.maxlen else self.env.maxlen
        self.gamma = gamma
        self.batch_size = batch_size
        loss = float("inf")
        no_improve_count = 0
        best_combined_metric = float("inf")

        for episode in range(self.num_episodes):
            print_progress_bar(episode + 1, self.num_episodes)
            self.env.reset()
            sum_rewards = 0
            epsilon = max(1 - episode / (self.num_episodes * 0.8), 0.01)

            for step in range(self.num_steps):
                state, action, reward, next_action, done = self.play_one_step(
                    x_data[step], y_data[step], epsilon
                )
                sum_rewards += reward if isinstance(reward, int) else reward[0]

                # Train if buffer has enough samples
                if len(self.replay_buffer) > self.batch_size:
                    loss = self._training_step()

                if done:
                    break

            combined_metric = loss - alpha * sum_rewards

            if combined_metric < best_combined_metric:
                best_combined_metric = combined_metric
                self.best_weights = self.model.get_weights()
                self.best_loss = loss
                no_improve_count = 0  # Reset counter on improvement
            else:
                no_improve_count += 1

            rewards.append(sum_rewards)

            # Logging
            if episode % (self.num_episodes // 10) == 0:
                print(
                    f"Episode: {episode}, Steps: {step+1}, Epsilon: {epsilon:.3f}, Loss: {loss:.2e}, Reward: {sum_rewards}, No Improve Count: {no_improve_count}"
                )

            # Early stopping condition
            if no_improve_count >= patience:
                print(
                    f"Early stopping at episode {episode} due to no improvement in {patience} episodes."
                )
                break

        # Save best model
        self.model.set_weights(self.best_weights)

    def __str__(self):
        return (
            f"AutoQL (Env: {self.env.name}, Episodes: {self.num_episodes}, Steps: {self.num_steps})"
        )


if __name__ == "__main__":
    pass
