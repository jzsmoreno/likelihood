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
        self.num_actions = [net[-1].out_features for net in self.action_networks]
        self.equal_action_sizes = len(set(self.num_actions)) == 1
        self.max_num_actions = max(self.num_actions)

    def forward(self, state: torch.Tensor, multiple_option: bool = False):
        """
        Parameters
        ----------
        state : torch.Tensor
            Tensor of shape (batch_size, state_dim) or (state_dim,)
        multiple_option : bool, default False
            Whether the model should return probabilities for multiple options.

        Returns
        -------
        option_probs : torch.Tensor
            Tensor of shape (batch_size, num_options)
        action_probs : torch.Tensor
            - If multiple_option=False: shape (batch_size, num_actions)
            - If multiple_option=True: shape (batch_size, num_options, num_actions)
        termination_prob : torch.Tensor
            Tensor of shape (batch_size, 1)
        selected_options : torch.Tensor
            Tensor of shape (batch_size,) indicating the selected option per batch item
        selected_actions : torch.Tensor
            - If multiple_option=False: shape (batch_size,)
            - If multiple_option=True: shape (batch_size, num_options)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.size(0)

        option_probs = torch.softmax(
            self.option_network(state), dim=-1
        )  # (batch_size, num_options)

        device = state.device
        num_options = len(self.action_networks)

        if multiple_option:
            action_probs = torch.zeros(batch_size, num_options, self.max_num_actions, device=device)
            selected_actions = torch.zeros(batch_size, num_options, dtype=torch.long, device=device)

            for i, net in enumerate(self.action_networks):
                probs = torch.softmax(net(state), dim=-1)
                num_actions_i = probs.size(-1)
                action_probs[:, i, :num_actions_i] = probs
                selected_actions[:, i] = torch.argmax(probs, dim=-1)

            selected_options = torch.argmax(option_probs, dim=-1)

        else:
            selected_options = torch.multinomial(option_probs, 1).squeeze(-1)
            action_probs = torch.zeros(batch_size, self.max_num_actions, device=device)
            selected_actions = torch.zeros(batch_size, dtype=torch.long, device=device)

            for opt_idx in torch.unique(selected_options):
                mask = selected_options == opt_idx
                if mask.any():
                    states_opt = state[mask]
                    probs = torch.softmax(self.action_networks[opt_idx](states_opt), dim=-1)
                    num_actions_i = probs.size(-1)
                    action_probs[mask, :num_actions_i] = probs
                    selected_actions[mask] = torch.argmax(probs, dim=-1)

        termination_prob = self.termination_network(state)

        return option_probs, action_probs, termination_prob, selected_options, selected_actions
