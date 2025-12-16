import warnings
from typing import List

import torch
import torch.nn as nn


class MultiBanditNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_options: int,
        num_actions_per_option: int | List[int],
        num_neurons: int = 128,
        num_layers: int = 1,
        activation: nn.Module = nn.ReLU(),
    ):
        super(MultiBanditNet, self).__init__()
        self.state_dim = state_dim
        self.num_options = num_options
        self.num_actions_per_option = num_actions_per_option
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.activation = activation

        self.option_network = nn.Sequential(
            nn.Linear(state_dim, self.num_neurons),
            nn.ReLU(),
            nn.Linear(
                self.num_neurons, num_options
            ),  # Output a probability distribution over options
        )

        # Low-level (action) Q-networks for each option with additional linear layers
        self.action_networks = nn.ModuleList()
        for i in range(num_options):
            action_network_layers = [nn.Linear(state_dim, self.num_neurons), self.activation]
            for _ in range(self.num_layers - 1):
                action_network_layers.extend(
                    [nn.Linear(self.num_neurons, self.num_neurons), self.activation]
                )
            num_actions = (
                num_actions_per_option
                if not isinstance(num_actions_per_option, list)
                else num_actions_per_option[i]
            )  # Output Q-values for each action in this option
            action_network_layers.append(nn.Linear(self.num_neurons, num_actions))
            self.action_networks.append(nn.Sequential(*action_network_layers))

        # Option termination network
        self.termination_network = nn.Sequential(
            nn.Linear(state_dim, self.num_neurons),
            nn.ReLU(),
            nn.Linear(self.num_neurons, 1),  # Single output for termination probability (0-1)
            nn.Sigmoid(),
        )

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        option_probs = torch.softmax(self.option_network(state), dim=-1)

        action_probs = []
        selected_actions = []

        for i in range(batch_size):
            selected_option = torch.multinomial(option_probs[i], 1).item()

            # Get Q-values for this option
            q_values = self.action_networks[selected_option](state[i].unsqueeze(0))
            action_prob = torch.softmax(q_values, dim=-1)
            action_probs.append(action_prob)
            selected_action = torch.argmax(action_prob, dim=-1)
            selected_actions.append(selected_action)

        if len(action_probs) > 0:
            action_probs = torch.cat(action_probs, dim=0).squeeze(1)
            selected_actions = torch.stack(selected_actions, dim=0).squeeze(1)
        else:
            warnings.warn(
                "The list of action probabilities is empty, initializing with default values.",
                UserWarning,
            )
            action_probs = torch.empty((batch_size, 1))
            selected_actions = torch.zeros(batch_size, dtype=torch.long)

        termination_prob = self.termination_network(state)

        return (
            option_probs,
            action_probs,
            termination_prob,
            torch.argmax(option_probs, dim=-1),  # selected_options
            selected_actions,
        )
