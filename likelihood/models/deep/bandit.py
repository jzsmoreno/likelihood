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
        activation: nn.Module = nn.SELU(),
        dropout_rate: float = 0.3,
    ) -> None:
        super(MultiBanditNet, self).__init__()
        self.state_dim = state_dim
        self.num_options = num_options
        self.num_actions_per_option = num_actions_per_option
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = max(0.0, dropout_rate)

        def build_action_network(num_actions: int) -> nn.Sequential:
            layers = [nn.Linear(state_dim, self.num_neurons), self.activation]
            for _ in range(max(0, self.num_layers - 1)):
                if self.dropout_rate > 0:
                    layers.extend(
                        [
                            nn.Dropout(self.dropout_rate),
                            nn.Linear(self.num_neurons, self.num_neurons),
                            self.activation,
                        ]
                    )
                else:
                    layers.extend([nn.Linear(self.num_neurons, self.num_neurons), self.activation])
            layers.append(nn.Linear(self.num_neurons, num_actions))
            return nn.Sequential(*layers)

        # Low-level (action) Q-networks for each option with additional linear layers
        self.action_networks = nn.ModuleList(
            build_action_network(
                num_actions_per_option[i]
                if isinstance(num_actions_per_option, list)
                else num_actions_per_option
            )
            for i in range(self.num_options)
        )

        def _dropout() -> nn.Module:
            return nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.option_network = nn.Sequential(
            nn.Linear(state_dim, self.num_neurons),
            self.activation,
            _dropout(),
            nn.Linear(
                self.num_neurons, num_options
            ),  # Output a probability distribution over options
        )

        # Option termination network
        self.termination_network = nn.Sequential(
            nn.Linear(state_dim, self.num_neurons),
            self.activation,
            _dropout(),
            nn.Linear(self.num_neurons, 1),  # Single output for termination probability (0-1)
            nn.Sigmoid(),
        )
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, self.num_neurons),
            self.activation,
            nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.num_neurons, 1),
        )
        self.num_actions = [net[-1].out_features for net in self.action_networks]
        self.equal_action_sizes = len(set(self.num_actions)) == 1
        self.max_num_actions = max(self.num_actions)

    def apply_initialization(self, method: str = "xavier_uniform") -> None:
        """
        Applies a specific initialization method to all linear layers in the network.

        Parameters
        ----------
        method : str, default "xavier_uniform"
            The initialization method to use. Supported methods:
            'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
            'orthogonal', 'uniform', 'normal', 'zeros', 'ones'
        """
        init_dispatch = {
            "xavier_uniform": nn.init.xavier_uniform_,
            "xavier_normal": nn.init.xavier_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
            "kaiming_normal": nn.init.kaiming_normal_,
            "orthogonal": nn.init.orthogonal_,
            "uniform": nn.init.uniform_,
            "normal": nn.init.normal_,
            "zeros": nn.init.zeros_,
            "ones": nn.init.ones_,
        }

        if method not in init_dispatch:
            raise ValueError(f"Unsupported initialization method: {method}")

        # Iterates over all modules (including submodules) and initializes linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_dispatch[method](m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, multiple_option: bool = False, temperature: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        state : torch.Tensor
            Tensor of shape (batch_size, state_dim) or (state_dim,)
        multiple_option : bool, default False
            Whether the model should return probabilities for multiple options.
        temperature : float, default 1.0
            Temperature used to control the sharpness of the output probability
            distribution. A value of ``1.0`` preserves the default behavior.
            Values greater than ``1.0`` produce a softer (more uniform)
            distribution, while values between ``0`` and ``1.0`` produce a
            sharper (more peaked) distribution. Must be strictly positive.

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
            self.option_network(state) / temperature, dim=-1
        )  # (batch_size, num_options)

        device = state.device
        num_options = len(self.action_networks)

        if multiple_option:
            action_probs = torch.zeros(batch_size, num_options, self.max_num_actions, device=device)
            selected_actions = torch.zeros(batch_size, num_options, dtype=torch.long, device=device)

            for i, net in enumerate(self.action_networks):
                probs = torch.softmax(net(state) / temperature, dim=-1)
                n_act = probs.size(-1)
                action_probs[:, i, :n_act] = probs
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
                    probs = torch.softmax(
                        self.action_networks[opt_idx](states_opt) / temperature, dim=-1
                    )
                    n_act = probs.size(-1)
                    action_probs[mask, :n_act] = probs
                    selected_actions[mask] = torch.argmax(probs, dim=-1)

        termination_prob = self.termination_network(state)

        return option_probs, action_probs, termination_prob, selected_options, selected_actions
