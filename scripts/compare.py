"""Compara dois experimentos (ex.: shaping de material puro vs. material+bônus
de lance) sobrepondo as curvas de recompensa, de lances legais por episódio e
de avaliação gulosa.

Uso:
    python -m scripts.compare --runs dqn_videochess dqn_move \
        --labels "material" "material+lance"
Gera ``results/comparacao.png``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ma(x, w=30):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = min(w, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def col(rows, name, default=0.0):
    return [float(r.get(name, default) or default) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    args = p.parse_args()
    labels = args.labels or args.runs
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, run in enumerate(args.runs):
        rows = read(RESULTS / run / "log.csv")
        ep = [int(r["episode"]) for r in rows]
        rew = col(rows, "ep_reward")
        moves = col(rows, "ep_moves")
        c = colors[i % len(colors)]

        m = ma(rew, 30)
        axes[0].plot(ep[len(ep) - len(m):], m, color=c, lw=2, label=labels[i])
        mv = ma(moves, 30)
        axes[1].plot(ep[len(ep) - len(mv):], mv, color=c, lw=2, label=labels[i])

        ev_path = RESULTS / run / "eval.csv"
        if ev_path.exists():
            ev = read(ev_path)
            axes[2].plot([int(r["global_step"]) for r in ev],
                         [float(r["eval_material"]) for r in ev],
                         "o-", color=c, label=labels[i])

    axes[0].set_title("Recompensa (média móvel 30) vs. Episódios")
    axes[0].set_xlabel("Episódio"); axes[0].set_ylabel("Recompensa")
    axes[1].set_title("Lances legais por episódio (média móvel 30)")
    axes[1].set_xlabel("Episódio"); axes[1].set_ylabel("nº de lances (board changes)")
    axes[2].set_title("Material na avaliação gulosa vs. Passos")
    axes[2].set_xlabel("Passo global"); axes[2].set_ylabel("Material (brancas - pretas)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("Comparação de funções de recompensa — Video Chess (Dueling DQN)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / "comparacao.png"
    fig.savefig(out, dpi=130)
    print("Figura salva em", out)


if __name__ == "__main__":
    main()
