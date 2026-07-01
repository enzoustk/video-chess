"""Gera as curvas de aprendizado a partir dos CSVs de treino.

Uso:
    python -m scripts.plot --run demo
Produz ``results/<run>/curvas.png``.
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


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < 1:
        return x
    w = min(w, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="demo")
    args = p.parse_args()
    run = RESULTS / args.run

    log = read_csv(run / "log.csv")
    ep = [int(r["episode"]) for r in log]
    reward = [float(r["ep_reward"]) for r in log]
    material = [float(r["ep_material"]) for r in log]
    eps = [float(r["epsilon"]) for r in log]
    loss = [float(r["mean_loss"]) for r in log]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(ep, reward, alpha=0.3, color="tab:blue", label="por episódio")
    ma = moving_average(reward, 50)
    ax.plot(ep[len(ep) - len(ma):], ma, color="tab:red", lw=2,
            label="média móvel (50)")
    ax.set_title("Recompensa (modelada) vs. Episódios")
    ax.set_xlabel("Episódio"); ax.set_ylabel("Recompensa do episódio")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, material, alpha=0.3, color="tab:green", label="por episódio")
    ma = moving_average(material, 50)
    ax.plot(ep[len(ep) - len(ma):], ma, color="tab:olive", lw=2,
            label="média móvel (50)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("Balanço material final vs. Episódios")
    ax.set_xlabel("Episódio"); ax.set_ylabel("Material (brancas - pretas)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ep, eps, color="tab:purple")
    ax.set_title("Exploração (epsilon) vs. Episódios")
    ax.set_xlabel("Episódio"); ax.set_ylabel("epsilon"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    eval_path = run / "eval.csv"
    if eval_path.exists():
        ev = read_csv(eval_path)
        if ev:
            steps = [int(r["global_step"]) for r in ev]
            er = [float(r["eval_reward"]) for r in ev]
            ax.plot(steps, er, "o-", color="tab:orange", label="avaliação gulosa")
            ax.set_title("Recompensa de avaliação (greedy) vs. Passos")
            ax.set_xlabel("Passo global"); ax.set_ylabel("Recompensa")
            ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.plot(ep, loss, color="tab:gray")
        ax.set_title("Perda (Huber) média vs. Episódios")
        ax.set_xlabel("Episódio"); ax.set_ylabel("loss")

    fig.suptitle(f"Curvas de aprendizado — run '{args.run}' (Video Chess, Dueling DQN)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = run / "curvas.png"
    fig.savefig(outpath, dpi=130)
    print("Figura salva em", outpath)


if __name__ == "__main__":
    main()
