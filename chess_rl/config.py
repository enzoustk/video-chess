"""Hiperparâmetros do agente e do treino (com valores padrão razoáveis)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # ----- ambiente / shaping -----
    seed: int = 0
    frameskip: int = 4
    sticky_prob: float = 0.0          # repeat_action_probability (0 facilita o cursor)
    max_episode_steps: int = 2000     # trunca partidas longas
    reward_mode: str = "material"     # "material" | "eval" (heurística + PST)
    material_scale: float = 0.1       # peso do Δ material (peão=0.1 ... dama=0.9)
    eval_scale: float = 0.1           # peso do shaping por avaliação heurística
    # γ do shaping por potencial: F = γ·Φ(s')−Φ(s). Usamos 1.0 (diferença pura
    # de potencial) para que posição estática => reward 0; γ<1 introduz um drift
    # por passo em posições estáticas (recompensaria "ficar parado" quando atrás).
    shaping_gamma: float = 1.0
    reward_clip: float = 1.0          # clipping da parte de shaping
    win_bonus: float = 1.0            # peso da recompensa nativa (vitória/derrota)
    step_penalty: float = 0.0
    move_bonus: float = 0.0           # bônus por lance legal (mudança no tabuleiro)
    # No mode=0 do Video Chess, o jogador humano padrão joga com as PRETAS
    # (o motor Atari joga com as brancas). Descoberta pela leitura do disassembly:
    # a rotina de seleção via FIRE rejeita peças com valor < 9 (as brancas).
    agent_white: bool = False

    # ----- DQN -----
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_freq: int = 4               # aprende a cada N passos de ambiente
    target_update_freq: int = 2_000   # atualização hard da rede-alvo
    grad_clip: float = 10.0           # clipping da norma do gradiente
    double_dqn: bool = True

    # ----- exploração (epsilon-greedy) -----
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    # NoisyNets: substitui ε-greedy por ruído paramétrico nas Linear da rede.
    # Quando True, ε é forçado a 0 (toda exploração vem do ruído da rede).
    use_noisy: bool = False
    noisy_sigma_init: float = 0.5

    # ----- treino / logging -----
    total_steps: int = 1_000_000
    eval_freq: int = 25_000
    eval_episodes: int = 5
    log_freq: int = 1_000             # frequência de flush do CSV
    checkpoint_freq: int = 50_000

    def to_dict(self):
        return asdict(self)
