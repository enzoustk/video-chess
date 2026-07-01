# Aprendizado por Reforço Profundo aplicado ao Atari *Video Chess*

Projeto final da disciplina **Aprendizado por Reforço** (UFRN / Metrópole Digital)
— Profª Tarciana Guerra. Implementação, treino e análise de um agente de **Deep RL**
(Dueling Double DQN) para o ambiente [`ALE/VideoChess-v5`](https://ale.farama.org/environments/video_chess/)
da *Arcade Learning Environment* (Farama Foundation).

> **Integrantes do grupo:** _(preencher)_

---

## 1. Visão geral

O *Video Chess* é, do ponto de vista de RL, um dos ambientes Atari **mais
difíceis**: a recompensa nativa é **extremamente esparsa** — só existe sinal
(+/−) ao fim de uma partida inteira de xadrez contra a IA do console, algo que
jogo aleatório **nunca** alcança (0 de recompensa em 5.000 passos aleatórios, sem
nenhum episódio terminar).

A contribuição central deste trabalho é o **projeto de uma função de recompensa
densa** (*reward shaping*). Descobrimos, por engenharia reversa, que **o tabuleiro
8×8 está codificado diretamente nos bytes 0–63 da RAM** do jogo. Isso permite
calcular o **balanço material** (peças brancas − peças pretas) a cada passo e
recompensar o agente por capturar peças adversárias e puni-lo por perder as suas:

```
r_t = clip(escala · Δmaterial, −c, +c) + bônus_vitória · r_nativo + penalidade_passo
```

Com isso, o problema deixa de ter curva de aprendizado plana em zero e passa a
ter um gradiente de recompensa denso, interpretável e treinável em um Mac (MPS).

## 2. Estrutura do repositório

```
video-chess/
├── chess_rl/                  # pacote principal
│   ├── board.py               # RAM -> tabuleiro e balanço material
│   ├── encoding.py            # RAM -> estado (planos 12×8×8 + auxiliares)
│   ├── env.py                 # make_env + wrapper de reward shaping
│   ├── network.py             # Dueling DQN híbrida (CNN do tabuleiro + MLP)
│   ├── replay.py              # Experience Replay (buffer circular)
│   ├── agent.py               # Double DQN + Huber loss + gradient clipping
│   └── config.py              # hiperparâmetros
├── scripts/
│   ├── probe_env.py           # inspeção/engenharia reversa da RAM
│   ├── train.py               # laço de treino + logging + checkpoints
│   ├── evaluate.py            # avaliação gulosa (+ gravação de vídeo)
│   └── plot.py                # curvas de aprendizado
├── relatorio/relatorio.md     # relatório técnico (5 objetivos)
├── slides/slides.md           # slides das apresentações
├── results/                   # logs, curvas e checkpoints (gerado)
└── requirements.txt
```

## 3. Instalação

```bash
# a partir da pasta video-chess/, usando o venv do repositório
source ../.venv/bin/activate          # ou crie: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # ale-py já traz a ROM do Video Chess
```

Requer Python ≥ 3.11. Em Apple Silicon o PyTorch usa **MPS** automaticamente;
em máquinas com GPU NVIDIA, usa CUDA; caso contrário, CPU.

## 4. Como usar

Todos os comandos são executados a partir da pasta `video-chess/`.

```bash
# (a) inspecionar o ambiente e a decodificação da RAM
python -m scripts.probe_env

# (b) treinar (run curto de demonstração, ~25-30 min em MPS)
python -m scripts.train --run dqn_videochess \
    --total-steps 400000 --eps-decay-steps 150000 \
    --learning-starts 5000 --eval-freq 25000 --eval-episodes 3

# (c) treino com sua HEURÍSTICA (material + piece-square tables) por potencial
python -m scripts.train --run dqn_eval --reward-mode eval --eval-scale 0.1 \
    --move-bonus 0.02 --total-steps 300000

# (c') TREINO CHESS-MOVE — agente escolhe LANCES DE XADREZ (Discrete(4096))
#     via cursor scriptado; recompensa por material ou heurística
python -m scripts.train_chess --run dqn_chess_mat --max-episodes 100 \
    --max-moves 30 --reward-mode material

# (d) treino com NOISYNETS (exploração paramétrica, sem ε-greedy)
python -m scripts.train --run dqn_noisy --use-noisy --reward-mode eval \
    --move-bonus 0.02 --total-steps 300000

# (e) MEDIR o agente (boletim de performance + classificação em níveis 0..3)
python -m scripts.benchmark --runs dqn_videochess dqn_move dqn_eval dqn_noisy \
    --episodes 6 --ckpt last.pt

# (f) engenharia reversa do RAM (F5=validade de lance) — ver seção 8
python -m scripts.probe_cursor --sweep-legal

# (g) gerar curvas comparativas A vs B vs C vs D
python -m scripts.plot --run dqn_noisy
python -m scripts.compare --runs dqn_videochess dqn_move dqn_eval dqn_noisy \
    --labels "A material" "B +lance" "C heurística" "D NoisyNets"

# (h) avaliar o agente treinado (política gulosa) e gravar vídeo

python -m scripts.evaluate --run dqn_noisy --episodes 5 --video --show-board

# (i) ASSISTIR o agente jogar ao vivo (janela em tempo real)
python -m scripts.watch --run dqn_noisy --ckpt best.pt --episodes 1 --fps 30
python -m scripts.watch --run dqn_noisy --ascii        # tabuleiro no terminal

# (j) VOCÊ jogar o Video Chess no teclado (contra a IA do Atari)
python -m scripts.play_human
```

> **Posso jogar *contra* o agente?** No Video Chess, o oponente é o **motor de
> xadrez embutido do Atari** (joga as pretas automaticamente) — não há um lugar
> para um segundo jogador humano. O agente de RL e você ocupam o **mesmo papel**
> (jogador das brancas vs. a IA do console). Então o que dá para fazer é
> **(i) assistir o agente jogar** e **(j) jogar você mesmo** contra a mesma IA
> que o agente enfrenta. Controles do modo humano: **setas** movem o cursor,
> **W/E/A/D** fazem diagonais, **ESPAÇO** = FIRE (seleciona/solta peça), **ESC** sai.

Saídas em `results/<run>/`: `config.json`, `log.csv`, `eval.csv`,
`curvas.png`, `best.pt`, `last.pt` e (opcional) `video/`.

## 5. Modelagem do problema (resumo)

| Componente | Definição |
|---|---|
| **Estado** | RAM (128 bytes) → tabuleiro one-hot **12×8×8** (6 tipos × 2 cores) + 64 bytes auxiliares (cursor, turno, UI) normalizados |
| **Ações** | `Discrete(10)`: NOOP, FIRE, 4 direções + 4 diagonais (move o cursor / seleciona peça) |
| **Recompensa** | opções: (a) Δ material clipado, (b) *reward shaping* baseado em potencial usando avaliação heurística (material + PST) — `reward_mode=eval` |
| **Algoritmo** | **Dueling Double DQN** + Experience Replay + rede-alvo + Huber + clipping; opcional **NoisyNets** (`--use-noisy`) para exploração paramétrica |

Detalhes completos e a análise crítica dos resultados estão em
[`relatorio/relatorio.md`](relatorio/relatorio.md).

## 6. Técnicas de estabilização e exploração implementadas

- **Experience Replay** (`replay.py`) — quebra a correlação temporal das amostras.
- **Rede-alvo** (*target network*) com atualização periódica — estabiliza o alvo do TD.
- **Double DQN** — reduz a superestimação dos valores Q.
- **Dueling architecture** — separa valor de estado e vantagem da ação.
- **Reward clipping** + **Huber loss** + **gradient norm clipping** — controlam a
  escala de recompensas e de gradientes.
- **Reward shaping baseado em potencial** (`evaluation.py`) — Φ(s) = material + PST;
  reward = γ·Φ(s')−Φ(s) (invariante à política, sem drift em posições estáticas).
- **NoisyNets** (`noisy.py`) — exploração paramétrica *state-dependent* (peça padrão
  do Rainbow DQN), substitui ε-greedy.

## 7. Métrica de performance — o "boletim" do agente

`scripts/benchmark.py` avalia agentes em eixos com significado de xadrez
(engajamento, material, avaliação heurística Φ, qualidade média/lance, taxa de
blunder, taxa de bons lances, término), sempre contra o **piso aleatório**, e
classifica o agente em um **nível** com o próximo gargalo a atacar:

| Nível | Descrição | Próximo passo |
|---|---|---|
| **0 — Inerte** | Não joga (congela) | Quebrar barreira de **exploração** |
| **1 — Ativo mas frágil** | Joga, perde material, blundera muito | Melhorar **qualidade** (heurística, n-step) |
| **2 — Competente** | Material equilibrado, poucos blunders | Aprender a **converter** vantagem em vitória |
| **3 — Vencedor** | Vence partidas contra a IA do Atari | Subir dificuldade do oponente |

## 8. Engenharia reversa do RAM do Video Chess (contribuição metodológica)

A partir da [documentação ALE](https://ale.farama.org/environments/video_chess/),
do [manual original AtariAge](https://atariage.com/manual_html_page.php?SoftwareLabelID=581)
e do [disassembly de Oscar Toledo G. em nanochess.org](https://nanochess.org/video_chess.html),
mapeamos os bytes chave do Video Chess. Rode `python -m scripts.probe_cursor --sweep-legal`
para reproduzir/validar.

| Byte (ALE) | Endereço 6507 | Papel |
|---|---|---|
| **0–63** | `$80–$BF` | **Tabuleiro 8×8**, um byte por casa; tipo da peça = `byte & 0x0F` (bits altos reaproveitados para outros dados) |
| **84** | `$D4` | *Source square* do lance corrente (= cursor no modo cursor / F3=0) |
| **85** | `$D5` | *Target square* do lance corrente (= cursor no modo seleção / F3=1) |
| **115** | `$F3` | Máquina de estados: `$00` cursor, `$01` selecionado, `$80` peça movendo, `$c0` engine pensando |
| **117** | `$F5` | **Validade do lance**: `0` = legal, `255` = ilegal |

**Encoding do cursor** (K=3): `byte = 3 + 8·(7 - rank) + file`, com wrap em 8 ranks
(byte percorre 3..59 em passos de 8). UP/DOWN alteram `±8` (rank), LEFT/RIGHT `±1`
(file). Debouncer de ~33 frames (sob `frameskip=4`, ~1 movimento a cada 4 taps).

**Achado crítico de correção**: nosso decoder inicial exigia bytes exatos
(70 = peão branco). Mas o Video Chess reusa os bits altos para outras finalidades
durante o jogo (cursor, castling flags, animação). Assim, uma casa com peão branco
podia aparecer como byte `6`, `70`, `102`, `198`... — todos com `& 0x0F == 6`.
`board.py` já usa a máscara correta; sem ela, o `material_balance` durante o jogo
era parcialmente espúrio.

**Trabalho futuro**: o 2º FIRE (release do lance) não completa o *commit* do lance
no engine (ram_F5 continua sinalizando algo inesperado após o FIRE em destino
legal). Resolver isso exige leitura linha-a-linha das rotinas de input no
disassembly do nanochess. Uma vez resolvido, `chess-bot` (via UCI) pode gerar
demonstrações no formato `(rank_origem, file_origem, rank_destino, file_destino)` →
cursor script → transições prontas para o replay buffer (**demonstration-based
bootstrapping**, o próximo salto de nível esperado).

## 9. Limitações e observações

O *Video Chess* exige que o agente aprenda primeiro a **meta-mecânica do cursor**
(navegar até uma peça, `FIRE`, navegar até o destino, `FIRE`) antes de capturar
qualquer peça. Mesmo com recompensa densa, capturas são eventos raros sob
exploração aleatória, o que produz curvas de aprendizado ruidosas/"serrilhadas".
O objetivo do trabalho **não** é vencer o xadrez, e sim demonstrar o pipeline
completo de Deep RL, projetar e comparar funções de recompensa, medir performance
com uma métrica interpretável e analisar criticamente o comportamento do agente.
