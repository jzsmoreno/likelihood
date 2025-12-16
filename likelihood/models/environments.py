from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


class ActionSpace:
    def __init__(self, num_actions):
        self._num_actions = num_actions

    @property
    def n(self):
        return self._num_actions


class OptionCriticEnv:
    """
    An environment for Option Critic reinforcement learning that processes a dataset of episodes.

    Attributes
    ----------
    episodes : `Dict[str, Dict]`
        Dataset of episodes with state, action, selected_option, reward, next_state, and done information.
    observation_space : `np.ndarray`
        Initial observation space shape (from first episode's state)
    done : `bool`
        Whether the current episode has terminated
    num_options : `int`
        Number of distinct options available in the dataset
    actions_by_option : `defaultdict(set)`
        Maps selected options to sets of actions that were taken with them
    unique_actions_count : `List[int]`
        Count of unique actions per option index (used for action space definition)
    action_space : `ActionSpace`
        Custom action space defined by unique actions per option
    idx_episode : `int`
        Current episode index being processed
    current_state : `np.ndarray`
        Current state observation in the environment
    """

    def __init__(
        self,
        episodes: Dict[int, Dict[str, List]],
    ):
        """
        Initializes the OptionCriticEnv with a dataset of episodes.

        Parameters
        ----------
        episodes : `Dict[int, Dict]`
            Dataset of episodes where keys are episode identifiers and values are episode data.
            Each episode must contain at least:
                - "state": List of state observations
                - "selected_option": List of selected options
                - "action": List of actions taken
                - "reward": List of rewards
                - "next_state": List of next states
                - "done": List of termination flags

        Raises
        ------
        ValueError
            If required fields ("state" or "selected_option") are missing from episode data
        """
        self.episodes = episodes

        required_keys = ["state", "action", "selected_option", "reward", "next_state", "done"]
        for episode_id, data in episodes.items():
            if not all(k in data for k in required_keys):
                raise ValueError(
                    f"Episode {episode_id} missing keys: {set(required_keys) - set(data.keys())}"
                )

        self.observation_space = np.array(episodes[0]["state"][0])
        self.done = False
        self.idx_episode = 0
        self.current_state = None
        self.num_options = len(set(episodes[0]["selected_option"]))
        self.actions_by_option = defaultdict(set)

        # Build fast lookup for transitions
        self.state_action_option_to_transition: Dict[Tuple, Dict[str, Any]] = {}

        for episode_id, data in episodes.items():
            states = data["state"]
            actions = data["action"]
            options = data["selected_option"]
            next_states = data["next_state"]
            rewards = data["reward"]
            dones = data["done"]

            for i in range(len(states)):
                state_key = tuple(states[i])
                key = (state_key, options[i], actions[i])

                self.state_action_option_to_transition[key] = {
                    "next_state": next_states[i],
                    "reward": rewards[i],
                    "done": dones[i],
                }

            for i, selected in enumerate(options):
                self.actions_by_option[selected].add(actions[i])

        self.unique_actions_count = [
            len(self.actions_by_option.get(i, set()))
            for i in range(max(self.actions_by_option.keys()) + 1)
        ]

        self.action_space = ActionSpace(self.unique_actions_count)

    def reset(self) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to a random episode and returns the initial state.

        Returns
        -------
        observation : `np.ndarray`
            Initial state observation
        info : `Dict`
            Empty dictionary (no additional information)
        """
        episode_id = np.random.choice(list(self.episodes.keys()))
        self.idx_episode = episode_id
        self.current_state = self.episodes[episode_id]["state"][0]
        return self.current_state, {}

    def step(self, action: int, option: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes an action with a specific option and returns the next state, reward, and termination status.

        Parameters
        ----------
        action : `int`
            Action index to execute
        option : `int`
            Selected option index

        Returns
        -------
        next_state : `np.ndarray`
            State after taking the action
        reward : `float`
            Immediate reward for the transition
        done : `bool`
            Whether the episode has terminated (from episode data)
        terminated : `bool`
            Whether the action-option pair was found in the dataset
        info : `Dict`
            Empty dictionary (no additional information)
        """
        key = (tuple(self.current_state), option, action)
        if key in self.state_action_option_to_transition:
            trans = self.state_action_option_to_transition[key]
            self.current_state = trans["next_state"]
            return trans["next_state"].copy(), trans["reward"], trans["done"], True, {}
        else:
            return self.current_state, 0.0, False, False, {}


if __name__ == "__main__":
    data = {
        0: {
            "state": [
                np.array([0.03503893, 0.0471871, 0.00121938, -0.00847874]),
                np.array([0.03598267, -0.14795232, 0.00104981, 0.28458866]),
            ],
            "selected_option": [0, 0],
            "action": [0, 0],
            "next_state": [
                np.array([0.03598267, -0.14795232, 0.00104981, 0.28458866]),
                np.array([0.03302363, -0.34308922, 0.00674158, 0.5776025]),
            ],
            "reward": [1.0, 1.0],
            "done": [False, False],
        },
        1: {
            "state": [
                np.array([0.04769269, -0.03987791, -0.01187594, 0.02884407]),
                np.array([0.04689513, -0.23482755, -0.01129905, 0.31775647]),
            ],
            "selected_option": [0, 0],
            "action": [0, 0],
            "next_state": [
                np.array([0.04689513, -0.23482755, -0.01129905, 0.31775647]),
                np.array([0.04219858, -0.42978677, -0.00494392, 0.6068548]),
            ],
            "reward": [1.0, 1.0],
            "done": [False, False],
        },
    }

    # Initialize environment
    env = OptionCriticEnv(episodes=data)
    env.reset()
    num_actions = env.action_space.n
    print("current state :", env.current_state)
    print("environment step :", env.step(1, 0))
    print("current state :", env.current_state)
    print("environment step :", env.step(1, 0))
    print("num_actions :", num_actions)
