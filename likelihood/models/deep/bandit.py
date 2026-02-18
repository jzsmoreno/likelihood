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

        # Option probabilities
        option_probs = torch.softmax(
            self.option_network(state), dim=-1
        )  # (batch_size, num_options)

        if multiple_option:
            padded_action_probs = []
            selected_actions_list = []

            for i, net in enumerate(self.action_networks):
                probs = torch.softmax(net(state), dim=-1)
                num_actions_i = probs.size(-1)

                if num_actions_i < self.max_num_actions:
                    padding = torch.zeros(
                        batch_size, self.max_num_actions - num_actions_i, device=state.device
                    )
                    probs = torch.cat([probs, padding], dim=-1)

                padded_action_probs.append(probs)
                selected_actions_list.append(torch.argmax(probs[:, :num_actions_i], dim=-1))

            # Stack to (batch_size, num_options, max_num_actions)
            action_probs = torch.stack(padded_action_probs, dim=1)
            selected_actions = torch.stack(
                selected_actions_list, dim=1
            )  # (batch_size, num_options)
            selected_options = torch.argmax(option_probs, dim=-1)
        else:
            selected_options = torch.multinomial(option_probs, 1).squeeze(-1)
            action_probs_list = []
            selected_actions_list = []

            for i in range(batch_size):
                net = self.action_networks[selected_options[i]]
                probs = torch.softmax(net(state[i].unsqueeze(0)), dim=-1).squeeze(0)
                num_actions_i = probs.size(-1)

                if num_actions_i < self.max_num_actions:
                    padding = torch.zeros(self.max_num_actions - num_actions_i, device=state.device)
                    probs = torch.cat([probs, padding], dim=-1)

                action_probs_list.append(probs)
                selected_actions_list.append(torch.argmax(probs[:num_actions_i]))

            action_probs = torch.stack(action_probs_list, dim=0)  # (batch_size, max_num_actions)
            selected_actions = torch.tensor(selected_actions_list, device=state.device)

        termination_prob = self.termination_network(state)  # (batch_size, 1)

        return (
            option_probs,
            action_probs,
            termination_prob,
            selected_options,
            selected_actions,
        )
