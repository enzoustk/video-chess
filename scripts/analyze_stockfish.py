"""Análise de qualidade dos lances via Stockfish (ACPL = Average Centipawn Loss).

Para cada lance do agente:
    - Pergunta ao Stockfish a avaliação da posição ANTES do lance.
    - Pergunta a avaliação DEPOIS do lance (agora é vez das brancas).
    - centipawn_loss = eval_before − (−eval_after)  [invertendo para lado de quem jogou]
      = quanto o lance piorou a posição na perspectiva das pretas.
    - Também compara com o melhor lance sugerido pelo Stockfish (perfect-move rate).

Métricas por partida:
    ACPL (média)         menor = melhor (0 = joga como Stockfish, >50 = amador)
    Best-move rate       fração de lances que coincidem com o top do Stockfish
    Blunder rate         % lances com centipawn_loss ≥ 200 (perde peça ou mais)
    Mistake rate         % lances com 100 ≤ loss < 200
    Inaccuracy rate      % lances com 50 ≤ loss < 100

Uso:
    python -m scripts.analyze_stockfish --run dqn_chess_mat_long --games 5 --depth 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from chess_rl.agent import get_device
from chess_rl.chess_env import N_ACTIONS, VideoChessMoveEnv, action_to_squares
from chess_rl.network import DuelingDQN
from chess_rl.scripted_moves import board_to_python_chess

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(path, device, noisy=False):
    net = DuelingDQN(N_ACTIONS, noisy=noisy).to(device)
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["online"])
    net.eval()
    return net


def act_greedy(net, ram, mask, device):
    from chess_rl.encoding import encode_batch
    b, a = encode_batch(np.asarray(ram)[None, :])
    with torch.no_grad():
        q = net(torch.from_numpy(b).to(device),
                torch.from_numpy(a).to(device)).cpu().numpy()[0]
    q[~mask] = -np.inf
    return int(q.argmax())


def eval_position(engine, board, depth: int) -> int:
    """Retorna avaliação em centipawns, na perspectiva do lado do turno."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(board.turn)
    if score.is_mate():
        # +inf se dá mate, -inf se toma mate
        return 10000 if score.mate() > 0 else -10000
    return score.score()


def best_move(engine, board, depth: int) -> chess.Move:
    return engine.play(board, chess.engine.Limit(depth=depth)).move


def analyze_game(env, policy_fn, engine, depth: int, max_moves: int, seed: int):
    """Joga uma partida e mede ACPL dos lances DAS PRETAS (nossos)."""
    ram, info = env.reset(seed=seed)
    board = board_to_python_chess(ram, black_to_move=True)
    losses = []
    matches = []      # se o nosso lance foi o melhor do Stockfish
    for step in range(max_moves):
        if not list(board.legal_moves):
            break
        # avaliação ANTES + melhor lance
        eval_before = eval_position(engine, board, depth)
        top = best_move(engine, board, depth)
        # decisão do agente
        mask = info["action_mask"]
        action = policy_fn(ram, mask)
        src, dst = action_to_squares(action)
        our_move = chess.Move(src, dst)
        # trate promoção: se peão chega no fim, python-chess exige promotion
        if board.piece_at(src) and board.piece_at(src).piece_type == chess.PAWN:
            rank = chess.square_rank(dst)
            if (board.turn == chess.BLACK and rank == 0) or (board.turn == chess.WHITE and rank == 7):
                our_move = chess.Move(src, dst, promotion=chess.QUEEN)
        # if our_move is not legal, skip (shouldn't happen with mask)
        if our_move not in board.legal_moves:
            continue
        # executa no env real
        ram, r, term, trunc, info = env.step(action)
        # após execução, engine Atari já jogou. Precisamos avaliar a posição APÓS
        # nosso lance (antes da resposta das brancas). Vamos usar python-chess:
        b_after_ours = board.copy()
        b_after_ours.push(our_move)
        eval_after_ours = eval_position(engine, b_after_ours, depth)
        # perda centipawn = eval antes - eval depois (perspectiva de quem jogou)
        # eval_before já é POV pretas; eval_after_ours é POV brancas (mudou o turno).
        # Para reverter: eval_after_ours_pov_pretas = -eval_after_ours
        loss = eval_before - (-eval_after_ours)
        loss = max(0, loss)  # ganho não conta como perda
        losses.append(loss)
        matches.append(our_move == top)
        # reconstroi position a partir da RAM real (inclui resposta das brancas)
        board = board_to_python_chess(ram, black_to_move=True)
        if term or trunc:
            break
    return losses, matches, info["material"]


def summarize(name: str, all_losses, all_matches, all_finalmats):
    losses = np.array([l for lst in all_losses for l in lst])
    matches = np.array([m for lst in all_matches for m in lst])
    acpl = float(np.mean(losses)) if len(losses) else 0.0
    best_rate = 100 * float(np.mean(matches)) if len(matches) else 0.0
    blunder = 100 * float(np.mean(losses >= 200)) if len(losses) else 0.0
    mistake = 100 * float(np.mean((losses >= 100) & (losses < 200))) if len(losses) else 0.0
    inacc = 100 * float(np.mean((losses >= 50) & (losses < 100))) if len(losses) else 0.0
    good = 100 * float(np.mean(losses < 50)) if len(losses) else 0.0
    mat = float(np.mean(all_finalmats))
    print(f"{name:24s} | ACPL={acpl:6.1f} | best-move={best_rate:5.1f}% | "
          f"good<50={good:4.1f}% inacc={inacc:4.1f}% mistake={mistake:4.1f}% "
          f"BLUNDER≥200={blunder:4.1f}% | mat final={mat:+5.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--max-moves", type=int, default=30)
    p.add_argument("--depth", type=int, default=10, help="profundidade Stockfish")
    p.add_argument("--device", default=None)
    p.add_argument("--stockfish", default="stockfish")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt_dir = RESULTS / args.run
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    noisy = cfg.get("use_noisy", False)
    net = load(ckpt_dir / "last.pt", device, noisy=noisy)
    print(f"Analisando '{args.run}' vs Stockfish@d{args.depth} — {args.games} partidas de até {args.max_moves} lances\n")

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        rng = np.random.default_rng(0)

        def random_policy(ram, mask):
            legals = np.where(mask)[0]
            return int(rng.choice(legals))

        def agent_policy(ram, mask):
            return act_greedy(net, ram, mask, device)

        # baseline
        print("== BASELINE aleatório-legal ==")
        env = VideoChessMoveEnv(max_moves=args.max_moves, reward_mode="material")
        rand_losses, rand_matches, rand_mats = [], [], []
        for g in range(args.games):
            L, M, m = analyze_game(env, random_policy, engine, args.depth, args.max_moves, 30000+g)
            rand_losses.append(L); rand_matches.append(M); rand_mats.append(m)
        summarize("aleatorio-legal", rand_losses, rand_matches, rand_mats)
        env.close()
        print()

        # agente
        print(f"== AGENTE {args.run} ==")
        env = VideoChessMoveEnv(max_moves=args.max_moves, reward_mode="material")
        ag_losses, ag_matches, ag_mats = [], [], []
        for g in range(args.games):
            L, M, m = analyze_game(env, agent_policy, engine, args.depth, args.max_moves, 30000+g)
            ag_losses.append(L); ag_matches.append(M); ag_mats.append(m)
        summarize(args.run, ag_losses, ag_matches, ag_mats)
        env.close()

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
