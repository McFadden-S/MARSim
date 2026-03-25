"""
Proximal Policy Optimization (PPO) agent for MARSim.

Contains:
    - ``PPO_Network``: Actor-critic neural network with configurable
      architecture (hidden layers, activation, dropout).
    - ``PPO``: Full PPO training loop with GAE advantage estimation,
      clipped policy/value losses, and learning-rate annealing.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
from tqdm import tqdm


def assign(data: dict, key: str, default=None, precondition: bool = True, critical: bool = False):
    """
    Look up *key* in *data* with a fallback to *default*.

    Args:
        data: Settings dictionary.
        key: Key to look up.
        default: Returned when *key* is absent and *critical* is False.
        precondition: Extra guard — if False, skip the lookup entirely.
        critical: If True and the key is missing, raise instead of defaulting.
    """
    if precondition and key in data:
        return data[key]
    if critical:
        raise KeyError(f"Missing critical value for key: {key}")
    return default


# ── Network ──────────────────────────────────────────────────────────────────

class PPO_Network(nn.Module):
    """
    Actor-critic network for discrete action spaces.

    The actor and critic share the same observation encoder architecture
    (configurable depth, width, activation, dropout) but maintain
    independent weights.  Output logits are clamped to [-20, 20] to
    prevent numerical overflow in the softmax.
    """

    ACTIVATION_FUNCTIONS = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU,
        "SiLU": nn.SiLU,
    }

    def __init__(self, agent_settings: dict, observation_shape: int, actions_shape: int):
        super().__init__()

        self.action_mask = None
        self.extracting = False

        # Deterministic action selection (for evaluation / ablation)
        self.deterministic_tr = assign(agent_settings, "deterministic_tr", default=False)
        self.deterministic_ts = assign(agent_settings, "deterministic_ts", default=False)

        # Architecture hyper-parameters
        hidden_layers = assign(agent_settings, "hidden-layers", 1)
        layer_size = assign(agent_settings, "layer-size", 64)
        activation_fn = assign(agent_settings, "activation-function", "Tanh")
        d_o = assign(agent_settings, "dropout", None)
        dropout_actor = d_o if d_o is not None else assign(agent_settings, "dropout-actor", 0.0)
        dropout_critic = d_o if d_o is not None else assign(agent_settings, "dropout-critic", 0.0)

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            """Orthogonal weight init with constant bias (PPO best practice)."""
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        input_size = int(np.array(observation_shape).prod())
        act_cls = self.ACTIVATION_FUNCTIONS[activation_fn]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, layer_size)),
            act_cls(),
            *sum(
                [
                    [layer_init(nn.Linear(layer_size, layer_size)), act_cls(), nn.Dropout(dropout_critic)]
                    for _ in range(hidden_layers)
                ],
                [],
            ),
            layer_init(nn.Linear(layer_size, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, layer_size)),
            act_cls(),
            *sum(
                [
                    [layer_init(nn.Linear(layer_size, layer_size)), act_cls(), nn.Dropout(dropout_actor)]
                    for _ in range(hidden_layers)
                ],
                [],
            ),
            layer_init(nn.Linear(layer_size, actions_shape), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the critic's value estimate for observations *x*."""
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action=None, mask=None):
        """
        Sample (or evaluate) an action and return (action, log_prob, entropy, value).

        Args:
            x: Observation tensor of shape [B, obs_dim].
            action: If provided, evaluate this action instead of sampling.
            mask: Optional boolean mask of invalid actions.
        """
        logits = self.actor(x)

        # Merge external mask with the persistent action_mask
        if mask is not None and self.action_mask is not None:
            mask = torch.logical_or(self.action_mask, mask)
        else:
            mask = mask if mask is not None else self.action_mask

        if mask is not None:
            logits = logits.clone()
            if mask.dim() == 1:
                logits[mask.repeat(x.shape[0], 1)] = -1e10
            else:
                logits[mask] = -1e10

        logits = torch.clamp(logits, -20, 20)
        probs = Categorical(logits=logits)

        if action is None:
            if self.training and not self.deterministic_tr:
                action = probs.sample()
            elif self.training and self.deterministic_tr:
                action = torch.argmax(probs.probs, dim=-1)
            else:
                action = torch.argmax(probs.probs, dim=-1) if self.deterministic_ts else probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_probs(self, x: torch.Tensor, action=None):
        """Return action probabilities, or log-probs for a given *action*."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is not None:
            return probs.log_prob(action)
        return probs.probs

    def get_next_likely_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the most likely action *other than* the one specified."""
        probs = self.get_probs(obs).clone()
        probs[torch.arange(probs.size(0)), action.long()] = 0
        probs = probs / probs.sum(dim=1, keepdim=True)
        return torch.argmax(probs, dim=1)

    def mask_actions(self, mask):
        """Set a persistent action mask applied to all future forward passes."""
        self.action_mask = mask


# ── PPO Algorithm ────────────────────────────────────────────────────────────

class PPO:
    """
    Complete PPO training wrapper.

    Handles inference (``step``), GAE advantage calculation, and the
    multi-epoch clipped update loop.  Model checkpoints can be saved
    and loaded with ``save`` / ``load``.

    Default hyper-parameters follow standard PPO recommendations:
        gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01,
        learning_rate=2.5e-4, update_epochs=4.
    """

    def __init__(self, observation_shape: int, action_shape: int,
                 device: str = "cpu", agent_settings: dict = {}):
        self.agent_settings = agent_settings
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.agent = PPO_Network(agent_settings, observation_shape, action_shape).to(device)

        # Environment
        self.num_envs = assign(agent_settings, "num-envs", 1)

        # GAE parameters
        self.gamma = assign(agent_settings, "gamma", 0.99)
        self.gae_lambda = assign(agent_settings, "gae-lambda", 0.95)

        # Optimisation parameters
        self.update_epochs = assign(agent_settings, "update-epochs", 4)
        self.num_minibatches = assign(agent_settings, "num-minibatches", None)
        self.minibatch_size = assign(agent_settings, "minibatch-size", 256)
        self.norm_adv = assign(agent_settings, "norm-adv", True)
        self.clip_coef = assign(agent_settings, "clip-coef", 0.2)
        self.clip_vloss = assign(agent_settings, "clip-vloss", True)
        self.vf_coef = assign(agent_settings, "vf-coef", 0.5)
        self.ent_coef = assign(agent_settings, "ent-coef", 0.01)
        self.target_kl = assign(agent_settings, "target-kl", None)

        # Optimizer
        self.anneal_lr = assign(agent_settings, "anneal-lr", True)
        self.learning_rate = assign(agent_settings, "learning-rate", 2.5e-4)
        self.weight_decay = assign(agent_settings, "weight-decay", 1e-4)
        self.optimizer_eps = assign(agent_settings, "optimizer-eps", 1e-5)
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_eps,
            weight_decay=self.weight_decay,
        )
        self.max_grad_norm = assign(agent_settings, "max-grad-norm", 0.5)

        self.device = device

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, desc: str):
        """Save model and optimizer state to ``{desc}.pth``."""
        torch.save(
            {"state_dict": self.agent.state_dict(), "optimizer": self.optimizer.state_dict()},
            f"{desc}.pth",
        )

    def load(self, desc: str):
        """Load model and optimizer state from ``{desc}.pth``."""
        checkpoint = torch.load(f"{desc}.pth")
        self.agent.load_state_dict(assign(checkpoint, "state_dict", critical=True))
        self.optimizer.load_state_dict(assign(checkpoint, "optimizer", critical=True))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, obs: torch.Tensor, action=None, mask=None):
        """
        Run a forward pass in inference mode (no gradients).

        Returns:
            (action, log_prob, entropy, value) — all detached tensors.
        """
        with torch.no_grad():
            action, logprob, entropy, value = self.agent.get_action_and_value(obs, action=action, mask=mask)
        return action, logprob, entropy, value.flatten()

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self, obs, actions, logprobs, rewards, dones, values, num_steps,
               last_observation, last_done, update, num_updates, is_nan_padding):
        """
        Perform a full PPO update from collected rollout data.

        1. Compute GAE advantages and returns.
        2. Flatten time-major rollout into a single batch.
        3. Run *update_epochs* of minibatch SGD with clipped losses.

        Args:
            obs:              [T, B, D] observations.
            actions:          [T, B] actions taken.
            logprobs:         [T, B] log-probabilities at collection time.
            rewards:          [T, B] rewards received.
            dones:            [T, B] done flags (float: 0.0 or 1.0).
            values:           [T, B] critic values at collection time.
            num_steps:        Total timesteps in the rollout.
            last_observation: [B, D] final observation for bootstrapping.
            last_done:        [B] final done flags.
            update:           Current update index (for LR annealing).
            num_updates:      Total planned updates (for LR annealing).
            is_nan_padding:   If True, NaN entries mark padding to ignore.
        """
        # Learning-rate annealing
        if self.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

        # --- GAE advantage calculation ---
        with torch.no_grad():
            next_value = self.agent.get_value(last_observation).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            last_gae_lam = 0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_non_terminal = 1.0 - last_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]

                delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam

            returns = advantages + values

        # --- Flatten rollout into batch ---
        if is_nan_padding:
            b_obs = obs[~torch.isnan(obs)].reshape((-1,) + (self.observation_shape,))
            del obs
            nan_mask = ~torch.isnan(actions.reshape(-1))
            b_logprobs = logprobs.reshape(-1)[nan_mask]
            b_actions = actions.reshape(-1)[nan_mask]
            b_values = values.reshape(-1)[nan_mask]
            b_advantages = advantages.reshape(-1)[nan_mask]
            b_returns = returns.reshape(-1)[nan_mask]
        else:
            b_obs = obs.reshape((-1,) + (self.observation_shape,))
            del obs
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_values = values.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

        del logprobs, actions, values, advantages, returns

        # --- Minibatch SGD ---
        batch_size = b_obs.shape[0]
        minibatch_size = (
            batch_size // self.num_minibatches
            if self.num_minibatches is not None
            else self.minibatch_size
        )

        batch_idx = np.arange(batch_size)
        approx_kl = None

        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_idx)

            for start in range(0, batch_size, minibatch_size):
                mb_idx = batch_idx[start:start + minibatch_size]

                _, new_log_prob, entropy, new_value = self.agent.get_action_and_value(
                    b_obs[mb_idx], b_actions.long()[mb_idx]
                )

                # Clipped log-ratio to prevent inf
                log_ratio = torch.clamp(new_log_prob - b_logprobs[mb_idx], min=-20, max=20)
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                # Advantage normalisation
                mb_adv = b_advantages[mb_idx]
                if self.norm_adv and mb_adv.numel() > 1:
                    std = mb_adv.std()
                    std = std if std > 1e-6 else 1.0
                    mb_adv = (mb_adv - mb_adv.mean()) / std

                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped)
                new_value = new_value.view(-1)
                if self.clip_vloss:
                    v_unclipped = (new_value - b_returns[mb_idx]) ** 2
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        new_value - b_values[mb_idx], -self.clip_coef, self.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_returns[mb_idx]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                # Total loss = policy + value + entropy bonus
                loss = pg_loss - self.ent_coef * entropy.mean() + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Early stopping on KL divergence
            if self.target_kl is not None and approx_kl is not None and approx_kl > self.target_kl:
                break
