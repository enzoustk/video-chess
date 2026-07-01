"""Boletim de performance do agente — métrica interpretável de "em que nível ele está".

Em vez de olhar só a recompensa modelada (que depende do shaping), este script
mede o agente em eixos com significado de xadrez, comparando sempre contra o
**baseline aleatório** (piso). Cada lance é atribuído ao agente (brancas) ou ao
oponente (pretas) assumindo alternância de turnos (brancas jogam primeiro), o
que permite medir a **qualidade dos lances do próprio agente**.

Eixos medidos (média por episódio):
  1. ENGAJAMENTO   -> nº de lances legais do agente (0 = congela)
  2. MATERIAL      -> balanço material final (brancas - pretas), em peões
  3. AVALIAÇÃO Φ   -> avaliação heurística final (material + PST), em peões
  4. QUALIDADE     -> ganho médio de avaliação por lance do agente (ΔΦ), em peões
  5. BLUNDERS      -> % dos lances do agente que pioram a posição em ≥ 2 peões
  6. TÉRMINO       -> % de partidas que terminaram (vitória/derrota nativa)

A partir disso, classifica um NÍVEL (0=Inerte ... 3=Vencedor) e aponta o
próximo gargalo a atacar.

Uso:
    python -m scripts.benchmark --runs dqn_videochess dqn_move dqn_eval --episodes 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chess_rl.agent import DQNAgent, get_device
from chess_rl.board import material_balance
from chess_rl.config import Config
from chess_rl.env import make_env
from chess_rl.evaluation import position_value

RESULTS = Path(__file__).resolve().parent.parent / "results"
BLUNDER_THRESHOLD = -2.0   # piora de avaliação (peões) que conta como blunder
GOODMOVE_THRESHOLD = 0.5   # melhora de avaliação (peões) que conta como bom lance


def run_policy(act_fn, cfg, episodes, seed_base=50_000):
    env = make_env(seed=cfg.seed, reward_mode="material",
                   agent_white=cfg.agent_white,
                   max_episode_steps=cfg.max_episode_steps,
                   frameskip=cfg.frameskip, sticky_prob=cfg.sticky_prob)
    eps_data = []
    for ep in range(episodes):
        ram, info = env.reset(seed=seed_base + ep)
        prev_board = np.asarray(ram)[:64].copy()
        prev_phi = position_value(ram, cfg.agent_white)
        n_changes = 0
        agent_d, opp_d = [], []
        fire = 0
        done = ended = False
        result = 0.0
        while not done:
            a = act_fn(ram)
            fire += int(a == 1)
            ram, r, term, trunc, info = env.step(a)
            board = np.asarray(ram)[:64]
            if not np.array_equal(board, prev_board):
                phi = position_value(ram, cfg.agent_white)
                d = phi - prev_phi
                # alternância: 0º,2º,... = lances do agente (brancas jogam 1º)
                (agent_d if n_changes % 2 == 0 else opp_d).append(d)
                n_changes += 1
                prev_phi = phi
                prev_board = board.copy()
            if term:
                ended = True
                result = info.get("native_reward", 0.0)
            done = term or trunc
        eps_data.append(dict(
            agent_moves=len(agent_d), fire=fire,
            material=material_balance(ram),
            phi_final=position_value(ram, cfg.agent_white),
            agent_d=agent_d, ended=ended, result=result,
        ))
    env.close()
    return aggregate(eps_data)


def aggregate(eps):
    all_agent_d = [d for e in eps for d in e["agent_d"]]
    n = max(1, len(all_agent_d))
    blunders = sum(1 for d in all_agent_d if d <= BLUNDER_THRESHOLD)
    good = sum(1 for d in all_agent_d if d >= GOODMOVE_THRESHOLD)
    ended = [e for e in eps if e["ended"]]
    wins = sum(1 for e in ended if e["result"] > 0)
    losses = sum(1 for e in ended if e["result"] < 0)
    return dict(
        engagement=float(np.mean([e["agent_moves"] for e in eps])),
        material=float(np.mean([e["material"] for e in eps])),
        phi=float(np.mean([e["phi_final"] for e in eps])),
        move_quality=float(np.mean(all_agent_d)) if all_agent_d else 0.0,
        blunder_rate=blunders / n,
        good_rate=good / n,
        completion_rate=len(ended) / len(eps),
        wins=wins, losses=losses,
        n_agent_moves=len(all_agent_d),
    )


def classify_level(m):
    # piso de engajamento: < 3 lances/episódio = não joga de forma consistente
    if m["engagement"] < 3.0:
        return (0, "INERTE / QUASE-INERTE — não joga de forma consistente (≈ congela).",
                "Próximo passo: quebrar a barreira de EXPLORAÇÃO "
                "(bootstrap por demonstração / NoisyNets / curiosidade).")
    # amostra mínima de lances para confiar nas métricas de qualidade
    if m["n_agent_moves"] < 12:
        return (1, "ATIVO INCIPIENTE — joga pouco; amostra insuficiente p/ qualidade.",
                "Próximo passo: aumentar ENGAJAMENTO consistente + mais treino.")
    if m["material"] < -1.0 or m["blunder_rate"] > 0.40:
        return (1, "ATIVO MAS FRÁGIL — joga, mas perde material / erra muito.",
                "Próximo passo: melhorar a QUALIDADE dos lances "
                "(reward shaping por avaliação, n-step, mais treino).")
    if m["wins"] == 0:
        return (2, "COMPETENTE — joga de forma equilibrada (material ~0, poucos blunders).",
                "Próximo passo: aprender a CONVERTER vantagem em vitória "
                "(treino mais longo, currículo, oponente mais fácil).")
    return (3, "VENCEDOR — vence partidas contra a IA do Atari.",
            "Próximo passo: subir o nível de dificuldade do oponente.")


def fmt_row(label, m):
    return (f"{label:22s} | eng {m['engagement']:5.1f} | mat {m['material']:+5.2f} "
            f"| Φ {m['phi']:+5.2f} | qual/lance {m['move_quality']:+5.2f} "
            f"| blunder {m['blunder_rate']*100:4.0f}% | bom {m['good_rate']*100:4.0f}% "
            f"| fim {m['completion_rate']*100:3.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--ckpt", default="last.pt", help="checkpoint a avaliar (last.pt = política final)")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = get_device(args.device)

    # baseline usa a config do 1º run (mesmo ambiente)
    base_cfg = Config(**{k: v for k, v in
                         json.loads((RESULTS / args.runs[0] / "config.json").read_text()).items()
                         if k in Config().to_dict()})

    rng = np.random.default_rng(0)
    n_actions = 10
    print(f"Rodando {args.episodes} episódios por política (device={device})...\n")

    results = {}
    print("== BASELINE ==")
    rnd = run_policy(lambda ram: int(rng.integers(0, n_actions)), base_cfg, args.episodes)
    results["aleatorio"] = rnd
    print(fmt_row("aleatório (piso)", rnd))

    print("\n== AGENTES ==")
    for run in args.runs:
        cfg = Config(**{k: v for k, v in
                        json.loads((RESULTS / run / "config.json").read_text()).items()
                        if k in Config().to_dict()})
        agent = DQNAgent(n_actions, cfg, device)
        agent.load(RESULTS / run / args.ckpt, map_location=device)
        m = run_policy(lambda ram: agent.act(ram, step=0, greedy=True), cfg, args.episodes)
        results[run] = m
        lvl, desc, nextstep = classify_level(m)
        print(fmt_row(run, m))
        print(f"    -> NÍVEL {lvl}: {desc}")
        print(f"       {nextstep}")
        (RESULTS / run / "benchmark.json").write_text(
            json.dumps({**m, "level": lvl, "level_desc": desc}, indent=2))

    print("\nLegenda: eng=lances do agente/episódio | mat=material final (peões) | "
          "Φ=avaliação heurística final | qual/lance=ΔΦ médio por lance do agente | "
          "blunder=lances que pioram ≥2 peões | bom=lances que melhoram ≥0,5 | fim=partidas terminadas")
    (RESULTS / "benchmark_resumo.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
