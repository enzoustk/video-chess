"""Treino do agente sobre o ChessMoveEnv — o espaço de ações é de LANCES DE
XADREZ (Discrete(4096)), com action masking para os lances legais das PRETAS.

Isso ataca o gargalo real detectado pela análise (atribuição de crédito
multi-passo do cursor) *pulando* a mecânica de cursor via ``execute_move`` +
``python-chess``. O agente RL agora aprende **xadrez de verdade** contra o
motor do Atari (brancas), com Dueling Double DQN sobre a heurística PST.

Uso:
    python -m scripts.train_chess --run dqn_chess --max-episodes 40 --max-moves 30
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chess_rl.agent import get_device
from chess_rl.chess_env import N_ACTIONS, VideoChessMoveEnv
from chess_rl.encoding import encode_batch
from chess_rl.network import DuelingDQN

RESULTS = Path(__file__).resolve().parent.parent / "results"


class ChessAgent:
    def __init__(self, device, lr=1e-4, gamma=0.99, hidden=256, noisy=False):
        self.device = device
        self.gamma = gamma
        self.online = DuelingDQN(N_ACTIONS, hidden=hidden, noisy=noisy).to(device)
        self.target = DuelingDQN(N_ACTIONS, hidden=hidden, noisy=noisy).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)

    def _tensors(self, ram_batch):
        b, a = encode_batch(ram_batch)
        return (torch.from_numpy(b).to(self.device),
                torch.from_numpy(a).to(self.device))

    @torch.no_grad()
    def act(self, ram, mask, eps, rng):
        legals = np.where(mask)[0]
        if len(legals) == 0:
            return 0   # no-op
        if rng.random() < eps:
            return int(rng.choice(legals))
        b, a = self._tensors(np.asarray(ram)[None, :])
        q = self.online(b, a).cpu().numpy()[0]
        q_masked = np.full_like(q, -np.inf)
        q_masked[legals] = q[legals]
        return int(q_masked.argmax())

    def learn(self, batch, grad_clip=10.0):
        """Double DQN + n-step (o `discount` no batch já é γ^n para cada amostra)."""
        ram, act_arr, rew_n, next_ram, done, next_mask, discount_n = batch
        b, a = self._tensors(ram)
        nb, na = self._tensors(next_ram)
        act_t = torch.as_tensor(act_arr, device=self.device).long()
        rew_t = torch.as_tensor(rew_n, device=self.device).float()
        done_t = torch.as_tensor(done, device=self.device).float()
        disc_t = torch.as_tensor(discount_n, device=self.device).float()
        next_mask_t = torch.as_tensor(next_mask, device=self.device).bool()

        q = self.online(b, a).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_online = self.online(nb, na)
            q_next_online = q_next_online.masked_fill(~next_mask_t, -1e9)
            next_actions = q_next_online.argmax(dim=1)
            q_next_target = self.target(nb, na).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            y = rew_t + disc_t * q_next_target * (1.0 - done_t)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), grad_clip)
        self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        torch.save({"online": self.online.state_dict(),
                    "target": self.target.state_dict()}, path)


class ChessBuffer:
    """Replay com n-step returns.

    Cada transição no buffer guarda o retorno n-step
    ``R_n = r + γ r' + γ² r'' + ... + γ^(n-1) r^(n-1)`` acumulado desde s_0
    até s_n (ou o fim do episódio, o que vier antes), e ``discount = γ^k``
    onde k é o número de passos efetivos (para casar com o bootstrap).
    """
    def __init__(self, cap: int, seed: int = 0, n_step: int = 1, gamma: float = 0.99):
        self.cap = cap
        self.n_step = n_step
        self.gamma = gamma
        self.ram = np.zeros((cap, 128), dtype=np.uint8)
        self.next_ram = np.zeros((cap, 128), dtype=np.uint8)
        self.actions = np.zeros(cap, dtype=np.int64)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.discounts = np.zeros(cap, dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.float32)
        self.next_mask = np.zeros((cap, N_ACTIONS), dtype=bool)
        self.pos = 0; self.size = 0
        self.rng = np.random.default_rng(seed)
        # buffer temporário para acumular n-step
        self._n_buf: list = []

    def _flush_n_step(self, terminal: bool = False):
        """Empurra transições n-step do buffer temporário para o replay."""
        while self._n_buf and (len(self._n_buf) >= self.n_step or terminal):
            n = min(self.n_step, len(self._n_buf))
            r0 = 0.0
            disc = 1.0
            for k in range(n):
                r0 += disc * self._n_buf[k][2]
                disc *= self.gamma
                if self._n_buf[k][4]:   # done dentro da janela
                    n = k + 1
                    break
            s0_ram, s0_act, _, _, _, _ = self._n_buf[0]
            _, _, _, sn_ram, sn_done, sn_mask = self._n_buf[n - 1]
            i = self.pos
            self.ram[i] = s0_ram
            self.actions[i] = s0_act
            self.rewards[i] = r0
            self.next_ram[i] = sn_ram
            self.dones[i] = float(sn_done)
            self.next_mask[i] = sn_mask
            self.discounts[i] = disc   # γ^n efetivo
            self.pos = (self.pos + 1) % self.cap
            self.size = min(self.size + 1, self.cap)
            self._n_buf.pop(0)
            if not terminal:
                break

    def add(self, ram, action, reward, next_ram, done, next_mask):
        self._n_buf.append((ram.copy(), action, reward, next_ram.copy(), done, next_mask.copy()))
        self._flush_n_step(terminal=done)
        if done:
            # descarrega o resto
            while self._n_buf:
                self._flush_n_step(terminal=True)

    def sample(self, batch):
        idx = self.rng.integers(0, self.size, size=batch)
        return (self.ram[idx], self.actions[idx], self.rewards[idx],
                self.next_ram[idx], self.dones[idx], self.next_mask[idx],
                self.discounts[idx])

    def __len__(self): return self.size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="dqn_chess")
    p.add_argument("--max-episodes", type=int, default=60)
    p.add_argument("--max-moves", type=int, default=30, help="max lances por episódio")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-episodes", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffer-size", type=int, default=20000)
    p.add_argument("--learning-starts", type=int, default=100)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--target-update-episodes", type=int, default=5)
    p.add_argument("--eval-scale", type=float, default=0.1)
    p.add_argument("--illegal-penalty", type=float, default=-0.5)
    p.add_argument("--use-noisy", action="store_true")
    p.add_argument("--reward-mode", default="eval", choices=["eval", "material"])
    p.add_argument("--n-step", type=int, default=1,
                   help="retornos n-step (1=Q-learning padrão, 3-5=comum em Rainbow)")
    args = p.parse_args()

    device = get_device(args.device)
    out = RESULTS / args.run; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"device": str(device)}, indent=2))
    print(f"[train_chess] run={args.run} device={device}")

    env = VideoChessMoveEnv(max_moves=args.max_moves, eval_scale=args.eval_scale,
                            illegal_penalty=args.illegal_penalty, seed=args.seed,
                            reward_mode=args.reward_mode)
    agent = ChessAgent(device, lr=args.lr, gamma=args.gamma, noisy=args.use_noisy)
    buf = ChessBuffer(args.buffer_size, seed=args.seed,
                      n_step=args.n_step, gamma=args.gamma)

    log_f = open(out / "log.csv", "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["episode", "steps", "reward_sum", "final_material",
                    "final_phi", "illegal_count", "eps", "mean_loss", "wall_sec"])

    rng = np.random.default_rng(args.seed)
    losses = deque(maxlen=100)
    total_steps = 0
    t0 = time.time()

    for ep in range(args.max_episodes):
        ram, info = env.reset()
        mask = info["action_mask"]
        ep_r = 0.0; ep_illegal = 0
        eps = args.eps_start + min(1.0, ep / max(1, args.eps_decay_episodes)) * (args.eps_end - args.eps_start)
        for step in range(args.max_moves):
            a = agent.act(ram, mask, eps if not args.use_noisy else 0.0, rng)
            next_ram, r, term, trunc, info = env.step(a)
            next_mask = info["action_mask"]
            buf.add(ram, a, r, next_ram, term, next_mask)
            ep_r += r
            if info.get("illegal"): ep_illegal += 1
            ram = next_ram; mask = next_mask
            total_steps += 1
            if len(buf) >= args.learning_starts and total_steps % args.train_freq == 0:
                losses.append(agent.learn(buf.sample(args.batch_size)))
            if term or trunc: break
        if (ep + 1) % args.target_update_episodes == 0:
            agent.update_target()
        wall = time.time() - t0
        mean_loss = float(np.mean(losses)) if losses else 0.0
        log_w.writerow([ep, step+1, round(ep_r, 3),
                        round(info["material"], 2), round(info["phi"], 2),
                        ep_illegal, round(eps, 3), round(mean_loss, 5), round(wall, 1)])
        log_f.flush()
        print(f"ep {ep:3d} steps={step+1:2d} R={ep_r:+6.2f} mat={info['material']:+.1f} "
              f"Φ={info['phi']:+.1f} illegal={ep_illegal} eps={eps:.2f} loss={mean_loss:.4f} {wall:.0f}s")

    agent.save(out / "last.pt")
    log_f.close(); env.close()
    print(f"[train_chess] concluído em {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
