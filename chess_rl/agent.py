"""Agente Double/Dueling DQN com rede-alvo, Huber loss e gradient clipping."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .encoding import encode_batch
from .network import DuelingDQN


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DQNAgent:
    def __init__(self, n_actions: int, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions
        self.online = DuelingDQN(n_actions, noisy=cfg.use_noisy,
                                 sigma_init=cfg.noisy_sigma_init).to(device)
        self.target = DuelingDQN(n_actions, noisy=cfg.use_noisy,
                                 sigma_init=cfg.noisy_sigma_init).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)

    # ----- codificação RAM -> tensores no device -----
    def _to_tensors(self, ram_batch):
        board, aux = encode_batch(ram_batch)
        board = torch.from_numpy(board).to(self.device)
        aux = torch.from_numpy(aux).to(self.device)
        return board, aux

    # ----- política epsilon-greedy (ou puramente NoisyNets) -----
    def epsilon(self, step: int) -> float:
        c = self.cfg
        if c.use_noisy:
            return 0.0   # exploração vem do ruído da rede, não do ε
        frac = min(1.0, step / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, ram, step: int, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon(step)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(0, self.n_actions))
        # NoisyNets: ruído fresco no modo treino; zerado em greedy
        if self.cfg.use_noisy:
            if greedy:
                self.online.eval_noise()
            else:
                self.online.reset_noise()
        board, aux = self._to_tensors(np.asarray(ram)[None, :])
        q = self.online(board, aux)
        return int(q.argmax(dim=1).item())

    # ----- passo de aprendizado (Double DQN + Huber + grad clip) -----
    def learn(self, batch) -> float:
        ram, actions, rewards, next_ram, dones = batch
        board, aux = self._to_tensors(ram)
        next_board, next_aux = self._to_tensors(next_ram)
        actions = torch.as_tensor(actions, device=self.device).long()
        rewards = torch.as_tensor(rewards, device=self.device).float()
        dones = torch.as_tensor(dones, device=self.device).float()
        # NoisyNets: ruído fresco a cada update, em ambas as redes
        if self.cfg.use_noisy:
            self.online.reset_noise()
            self.target.reset_noise()

        q = self.online(board, aux).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                next_actions = self.online(next_board, next_aux).argmax(dim=1)
                next_q = self.target(next_board, next_aux).gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target(next_board, next_aux).max(dim=1).values
            target = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q, target)  # Huber loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(),
                                           self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    # ----- checkpoint -----
    def save(self, path):
        torch.save({"online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "config": self.cfg.to_dict()}, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)

        def remap(sd):
            # compat com checkpoints antigos que salvavam o trunk como Sequential
            new = {}
            for k, v in sd.items():
                if k == "trunk.0.weight":
                    new["trunk_lin.weight"] = v
                elif k == "trunk.0.bias":
                    new["trunk_lin.bias"] = v
                else:
                    new[k] = v
            return new

        self.online.load_state_dict(remap(ckpt["online"]), strict=False)
        self.target.load_state_dict(remap(ckpt["target"]), strict=False)
