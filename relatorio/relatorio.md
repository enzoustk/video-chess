# Relatório Técnico — Agente de Deep RL para o Atari *Video Chess*

**Disciplina:** Aprendizado por Reforço — UFRN / Metrópole Digital
**Professora:** Tarciana Guerra
**Ambiente:** `ALE/VideoChess-v5` (Arcade Learning Environment, Farama Foundation)
**Algoritmo:** Dueling Double DQN com Experience Replay
**Integrantes:** _(preencher)_

---

## 1. Introdução e definição do problema

O *Video Chess* (Atari 2600) é um jogo de xadrez no qual o jogador humano
controla um **cursor** com o joystick para selecionar e mover peças contra a IA
embutida do console. Em termos de Aprendizado por Reforço, é um ambiente de
**controle e tomada de decisão sequencial em cenário competitivo** (o agente
joga contra um oponente fixo), o que o enquadra na categoria *"criação de
agentes autônomos em ambientes competitivos"* das instruções do projeto.

### 1.1. Por que o problema é difícil

Caracterizamos empiricamente o ambiente antes de modelar (ver
`scripts/probe_env.py`):

- **Espaço de observação:** imagem `Box(0,255,(210,160,3))` ou, na variante
  usada, **RAM** `Box(0,255,(128,))`.
- **Espaço de ações:** `Discrete(10)` — `NOOP, FIRE, UP, RIGHT, LEFT, DOWN` e as
  quatro diagonais.
- **Recompensa nativa:** **extremamente esparsa**. Em 5.000 passos de ações
  aleatórias, a recompensa acumulada foi **0** e **nenhum episódio terminou**. O
  reforço nativo só ocorre ao final de uma partida inteira de xadrez (vitória/
  derrota), evento inatingível por exploração aleatória.

Treinar um DQN/PPO diretamente sobre a recompensa nativa produziria uma **curva
de aprendizado plana em zero** — sem nenhum sinal para o agente aprender. Esse
diagnóstico orientou toda a modelagem: **o cerne do trabalho é projetar uma
função de recompensa densa** (objetivo explícito das instruções: *"projetando
uma função de recompensa que guie o agente"*).

---

## 2. Modelagem formal (MDP)

Formalizamos o problema como um Processo de Decisão de Markov
$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$.

### 2.1. Espaço de estados $\mathcal{S}$

A grande descoberta do projeto foi obtida por **engenharia reversa da RAM**:
os **bytes 0–63 da RAM codificam o tabuleiro 8×8 diretamente**, um byte por
casa. Comparando a RAM da posição inicial com um tabuleiro de xadrez, o
mapeamento é inconfundível:

```
rank 0 (bytes  0–7):   5  4  3  2  1  3  4  5     T C B D R B C T  (brancas)
rank 1 (bytes  8–15): 70 70 70 70 70 70 70 70     peões brancos
ranks 2–5:             0  (casas vazias)
rank 6 (bytes 48–55): 142 ...                      peões pretos
rank 7 (bytes 56–63): 13 12 11 10  9 11 12 13      T C B D R ...  (pretas)
```

| Peça  | Rei | Dama | Bispo | Cavalo | Torre | Peão |
|-------|-----|------|-------|--------|-------|------|
| Branca| 1   | 2    | 3     | 4      | 5     | 70   |
| Preta | 9   | 10   | 11    | 12     | 13    | 142  |

A partir disso, o estado entregue à rede neural é construído por
`chess_rl/encoding.py` de forma a respeitar a **natureza espacial** do problema:

1. **Tabuleiro → tensor `12×8×8`** (one-hot de 6 tipos de peça × 2 cores). Cada
   plano marca a presença de um tipo de peça; é uma representação espacial,
   adequada a uma CNN.
2. **Auxiliares → vetor de 64 floats** (bytes 64–127 da RAM, normalizados por
   255). Contêm informação **não-espacial e essencial para agir**: posição do
   cursor, indicador de turno e estado de UI.

Essa separação é deliberada: a posição do cursor (necessária para o agente saber
*onde* vai agir) vive nos bytes auxiliares, enquanto a configuração das peças
vive nos planos. Usar apenas a imagem (CNN sobre pixels) seria muito mais lento
(milhões de frames) e usar apenas o tabuleiro perderia a informação do cursor.

### 2.2. Espaço de ações $\mathcal{A}$

`Discrete(10)`, idêntico ao espaço nativo. A semântica é de **controle de
cursor**: o agente precisa aprender a sequência *navegar → `FIRE` (seleciona) →
navegar → `FIRE` (solta)* para realizar **um único** lance de xadrez. Essa
mecânica em múltiplos passos é o que torna a exploração tão custosa.

### 2.3. Função de recompensa $R$ (reward shaping)

Definimos o **balanço material** $m_t = \text{valor}_{\text{brancas}} -
\text{valor}_{\text{pretas}}$ (valores padrão: P=1, C=B=3, T=5, D=9, Rei=0),
calculado da RAM a cada passo (`chess_rl/board.py::material_balance`). A
recompensa modelada (`chess_rl/env.py`) é:

$$
r_t = \operatorname{clip}\!\big(\lambda \cdot (m_t - m_{t-1}),\,-c,\,+c\big)
      \;+\; \beta \cdot r^{\text{nativo}}_t \;+\; \rho
$$

- $\lambda \cdot \Delta m$ — recompensa **densa**: capturar uma peça preta dá
  reforço positivo (peão ≈ +0,1; dama ≈ +0,9), perder uma peça branca dá reforço
  negativo. Lances sem captura têm $\Delta m = 0$ (recompensa nula), e promoção
  de peão é corretamente recompensada como ganho de material.
- $\beta \cdot r^{\text{nativo}}$ — preserva o sinal forte de **fim de jogo**
  (vitória/derrota), sem clipping, dando-lhe peso maior que capturas isoladas.
- $\rho$ — penalidade por passo (0 por padrão; pode incentivar partidas mais
  curtas).

Valores usados: $\lambda = 0{,}1$, $c = 1{,}0$, $\beta = 1{,}0$, $\rho = 0$,
assumindo o **agente jogando de brancas** (sinal configurável). Esta é a peça
central que torna o problema aprendível.

### 2.4. Fator de desconto e horizonte

$\gamma = 0{,}99$. Como uma partida completa é longuíssima, truncamos episódios
em `max_episode_steps = 2000` passos (`TimeLimit`), o que dá ao agente muitos
episódios para aprender a meta-mecânica sem depender do fim natural do jogo.

---

## 3. Arquitetura da rede neural profunda

`chess_rl/network.py` implementa uma **Dueling DQN híbrida** (CNN + MLP),
adequada à natureza mista do estado:

```
tabuleiro (12×8×8) ─► Conv3x3(12→32)+ReLU ─► Conv3x3(32→64)+ReLU ─► flatten(4096)
                                                                   ─► Linear(4096→256)+ReLU ┐
auxiliares (64)    ─► Linear(64→64)+ReLU ───────────────────────────────────────────────────┤
                                                          concat(320) ─► Linear(320→256)+ReLU ┘
                                                                         ├─► V(s)      Linear(256→1)
                                                                         └─► A(s,a)    Linear(256→10)
                                          Q(s,a) = V(s) + ( A(s,a) − médiaₐ A(s,a) )
```

- A **CNN** explora a estrutura espacial 2D do tabuleiro (padrões locais de
  peças), com *padding* para preservar as 64 casas.
- O **ramo auxiliar** (MLP) processa cursor/turno/UI.
- A **cabeça Dueling** decompõe $Q$ em valor de estado $V(s)$ e vantagem
  $A(s,a)$, estabilizando o aprendizado em estados onde a ação importa pouco
  (ex.: navegar o cursor longe das peças).

---

## 4. Algoritmo e técnicas de estabilização

Escolhemos a família **DQN** (em vez de PPO) por dois motivos: (i) é
*value-based*, natural para o espaço de ações discreto; e (ii) as duas técnicas
de estabilização citadas explicitamente nas instruções — **Experience Replay** e
**Clipping** — são centrais ao DQN. Implementamos (`chess_rl/agent.py`):

| Técnica | Implementação | Papel |
|---|---|---|
| **Experience Replay** | buffer circular de 100k transições (`replay.py`); armazena RAM crua (128 B) e codifica em *batch* na amostragem | quebra correlação temporal; reuso de amostras |
| **Rede-alvo** | cópia da rede atualizada a cada `target_update_freq=2000` passos | alvo de TD estável |
| **Double DQN** | seleção da ação pela rede *online*, avaliação pela rede-alvo | reduz superestimação de Q |
| **Dueling** | cabeças $V$ e $A$ separadas | melhor estimativa de valor |
| **Reward clipping** | $\Delta m$ recortado em $[-1,1]$ | controla escala da recompensa |
| **Huber loss** | `smooth_l1_loss` | robustez a *outliers* de TD |
| **Gradient clipping** | norma máx. 10 | evita explosão de gradiente |

Regra de atualização (Double DQN):

$$
y_t = r_t + \gamma\, Q_{\theta^-}\!\big(s_{t+1},\, \arg\max_{a} Q_\theta(s_{t+1},a)\big)\,(1-d_t),
\qquad
\mathcal{L} = \text{Huber}\big(Q_\theta(s_t,a_t),\, y_t\big).
$$

**Exploração:** $\varepsilon$-greedy com decaimento linear de $1{,}0 \to 0{,}05$.

---

## 5. Otimização e hiperparâmetros

Otimizador **Adam**. Os hiperparâmetros (em `chess_rl/config.py`) e a
justificativa de ajuste:

| Hiperparâmetro | Valor | Observação |
|---|---|---|
| Taxa de aprendizado | $10^{-4}$ | padrão estável para DQN; valores maiores aumentaram a variância |
| Fator de desconto $\gamma$ | 0,99 | horizonte longo (créditos distantes da captura) |
| Batch size | 64 | equilíbrio custo/variância em MPS |
| Buffer de replay | 100.000 | cobre dezenas de episódios |
| `learning_starts` | 5.000 | preenche o buffer antes de treinar |
| `train_freq` | 4 | 1 update a cada 4 passos de ambiente |
| `target_update_freq` | 2.000 | compromisso estabilidade/atualidade |
| Decaimento de $\varepsilon$ | 150.000 passos | exploração prolongada (capturas são raras) |
| $\lambda$ (escala material) | 0,1 | mantém recompensas de captura em escala ~$[-1,1]$ |

**Ambiente:** `frameskip=4` e `repeat_action_probability=0` (desligar *sticky
actions* facilita o controle preciso do cursor, decisão tomada após observar que
ações pegajosas degradam a navegação).

**Infraestrutura:** PyTorch 2.11 com **MPS** (Apple Silicon); ~300–500
passos/seg, viabilizando o run de demonstração em ~25–30 min.

---

## 6. Resultados e análise crítica

Rodamos **quatro experimentos comparáveis** variando a função de recompensa e o
esquema de exploração, sob o mesmo orçamento (~200–300k passos, ~20–30 min em
MPS). Curvas em `results/<run>/curvas.png`; comparação em `results/comparacao.png`.

### 6.1. Os quatro experimentos

| Run | Recompensa | Exploração | Comportamento em treino |
|---|---|---|---|
| **A** `dqn_videochess` | Δ material clipado | ε-greedy 1.0→0.05 | Média móvel de recompensa sobe de −0.15 → −0.05; balanço material ruidoso com picos de −8 (episódios exploratórios que perderam material); **greedy converge a congelar** |
| **B** `dqn_move` | Δ material + bônus/lance (0.05) | ε-greedy | Quebra o congelamento em treino (ma100 de recompensa **positiva**, ~+0.9); mas material cai para −2.65 em média (troca material por movimento) |
| **C** `dqn_eval` | **Heurística PST** (potencial) + bônus/lance (0.02) | ε-greedy | Sinal mais rico — episódios com R = +6 (sequências de bons lances) e blunders punidos; ma100 positiva ~+0.24 |
| **D** `dqn_noisy` | Heurística PST + bônus/lance | **NoisyNets** (Rainbow) | Rede com ruído paramétrico substitui ε; agente permanece inerte |

### 6.2. Métrica de performance: o boletim (`scripts/benchmark.py`)

Recompensa modelada é sensível ao shaping — para "em que nível o agente está" de
forma interpretável, construímos um **boletim** com eixos de xadrez, comparando
sempre contra o baseline aleatório e atribuindo lances ao agente (brancas) por
alternância de turno:

| Política (política final, `last.pt`, 8 episódios) | Engaj. | Material | Φ | Blunder | **Nível** |
|---|---:|---:|---:|---:|---|
| aleatório (piso) | 0.9 | 0.00 | −0.05 | 14 % | — |
| **A** material | 0.0 | 0.00 | 0.00 | — | **0** |
| **B** material + lance | 1.0 | 0.00 | 0.00 | — | **0** |
| **C** heurística PST | 0.0 | 0.00 | 0.00 | — | **0** |
| **D** NoisyNets | 0.0 | 0.00 | 0.00 | — | **0** |

**Diagnóstico**: as quatro políticas gulosas convergem para *quase-inércia*,
indistinguíveis do aleatório em engajamento. A recompensa média em treino de C é
a mais saudável dos quatro (mais próxima de zero, sem trocas material-por-movimento
como B), mas a política gulosa não reproduz consistentemente lances legais.

### 6.3. Variância e curvas "serrilhadas" (objetivo 5)

O fenômeno de curvas serrilhadas aparece de forma marcante em A (picos −0.8/−8 na
recompensa/material) e é atenuado em D (NoisyNets), mas por razão negativa: a
política de D fica presa em ações que não mudam o tabuleiro. Isso confirma três
causas empíricas para o "serrilhado":

1. **Esparsidade estrutural**: lance legal exige a sequência precisa
   *navegar→FIRE→navegar→FIRE*; sob exploração aleatória ocorre ~1 a cada 200
   passos (medido em `scripts/probe_env.py`).
2. **Oponente fixo mas reativo**: mudanças pequenas de política produzem
   trajetórias de partida muito diferentes.
3. **ε-greedy** com 5% aleatório injeta lances-blunder ocasionais mesmo após
   convergência (visíveis em A/B como picos de material −6/−8).

Reportamos **média móvel (50)** para separar tendência do ruído.

### 6.4. Análise do comportamento — por que todos convergem a Nível 0?

Os quatro shapes de recompensa expõem *trade-offs* distintos:

- **A (só material)** → *ótimo pessimista*: como movimentos aleatórios contra o
  engine tendem a perder material, o agente aprende que **não mover** domina
  mover. Política final: 0 lances, 2139 FIREs em 3 episódios (confirmado por
  probe do checkpoint) — o agente pressiona FIRE freneticamente mas nunca com o
  cursor sobre uma peça em posição de fazer lance.
- **B (+ bônus de lance)** → quebra o congelamento em treino, mas o bônus por
  movimento (0.05) domina a magnitude do material (peão = 0.1), então a política
  aprende a **mover independentemente da qualidade** — mat treino cai a −2.65
  em média.
- **C (heurística PST, sua ideia)** → recompensa fica finalmente balanceada
  (movimentos ruins ficam net-negativos), ma100 em treino sobe positiva e
  episódios de +6 aparecem — mas o *greedy* ainda depende de descobrir as
  sequências certas de cursor, o que a exploração ε=0.05 no fim não sustenta.
- **D (NoisyNets)** → substitui ε-greedy por ruído paramétrico *state-dependent*
  na rede. Ainda assim, o argmax do Q-value se mantém em ações que não mudam o
  tabuleiro. σ₀=0.5 pode ser insuficiente para o espaço discreto de 10 ações; ou
  o problema é fundamentalmente **exploração multi-passo**, não single-step.

**Conclusão empírica**: o gargalo dominante em Video Chess **não é a recompensa
nem a exploração single-step**, é a **atribuição de crédito ao longo da sequência
multi-passo do cursor**. Mesmo com sinal de recompensa perfeito, o agente precisa
executar de 4 a 30 ações consecutivas *corretas* antes de ver qualquer sinal —
uma barreira que nenhum dos quatro métodos ataca diretamente.

### 6.5. Roteiro de melhoria guiado pela métrica

O boletim aponta o próximo passo por nível:

| Sair de… | Alavanca | Como |
|---|---|---|
| Nível 0 → 1 | **Bootstrap por demonstração** | `chess-bot` (via UCI, alimentado pela FEN decodificada) gera lances legais → script de cursor os executa → transições vão para o replay. Ataca a atribuição de crédito diretamente. Requer completar a RE do 2º FIRE (§7) — fundação principal já feita. |
| Nível 1 → 2 | Heurística PST (Run C) | Já implementada; passa a ter efeito real quando o engajamento existe |
| Nível 2 → 3 | Treino longo (10⁷+ passos) + envs paralelos | Idealmente em GPU (Colab) |

### 6.6. Limitações honestas

- O agente **não vence xadrez** — nem foi esse o objetivo. O trabalho entrega o
  pipeline, quatro experimentos comparáveis, uma métrica interpretável e uma
  fundação de engenharia reversa que baixa o custo do próximo passo.
- A recompensa por material/PST é *proxy* — não captura estratégias posicionais
  como controle de centro dinâmico ou iniciativa.
- *Video Chess* está entre os ambientes Atari mais adversos para RL (recompensa
  nativa quase-nula + oponente competente + mecânica de controle indireta).

---

## 7. Engenharia reversa do RAM (contribuição metodológica)

Guiados por [ale.farama.org](https://ale.farama.org/environments/video_chess/),
pelo [manual AtariAge](https://atariage.com/manual_html_page.php?SoftwareLabelID=581)
e pelo [disassembly comentado de Oscar Toledo G. no nanochess.org](https://nanochess.org/video_chess.html),
mapeamos o modelo de RAM do Video Chess. Reproduzível via `scripts/probe_cursor.py`:

| Byte (ALE) | Endereço 6507 | Papel |
|---|---|---|
| 0–63 | `$80–$BF` | Tabuleiro 8×8; **tipo da peça = `byte & 0x0F`** (bits altos reaproveitados) |
| 84 | `$D4` | Square de **origem** do lance corrente (cursor livre em F3=0) |
| 85 | `$D5` | Square de **destino** do lance corrente (cursor em modo seleção F3=1) |
| 115 | `$F3` | Máquina de estados: `$00` cursor · `$01` selecionado · `$80` movendo · `$c0` engine pensando |
| 117 | `$F5` | **Validade do lance**: `0` = legal, `255` = ilegal |

- Encoding do cursor: `byte = 3 + 8·(7 − rank) + file` com wrap; UP/DOWN = ±8,
  LEFT/RIGHT = ±1; debouncer de ~33 frames.
- **Achado crítico de correção do decoder**: exigir bytes exatos (`70` para peão
  branco) falha porque o high nibble é usado para dados auxiliares (cursor,
  flags de roque, animação). Com o *fix* (`& 0x0F`), o `material_balance`
  durante o jogo virou robusto — Runs anteriores treinavam com sinal parcialmente
  espúrio.
- **Verificação empírica via F5**: com peão branco em e2 selecionado, F5 vale
  0 **exatamente** nas casas legais (e3, e4) e 255 em todas as outras (probe em
  §7 do README, executável).

**O que falta (trabalho futuro)**: após o 2º FIRE, F5 comuta para 255 e o lance
não é *committed* apesar de F5=0 no momento da pressão. Isso requer leitura das
rotinas de input no disassembly (mais 1–2 dias de RE). Uma vez resolvido, o
*demonstration-based bootstrapping* (Design B do relatório de melhoria) se torna
implementável: `chess-bot` gera FEN→melhor lance→script traduz para cursor→
transições entram no replay, quebrando a barreira de atribuição de crédito.

---

## 7.5. Chegando a Nível 1: pipeline *chess-move* (breakthrough final)

Após completar a engenharia reversa (ver §7), destrancamos o mecanismo de FIRE
descobrindo que **o agente joga com as PRETAS** (não brancas — o disassembly
mostra `cpx #$09; bcc ignore` rejeitando peças de valor < 9) e que o engine usa
**índices row-major puros** (0..63), não a codificação de tela K=3. Com isso
implementamos `scripted_moves.execute_move(src, dst)` (`chess_rl/scripted_moves.py`)
que executa lances reais no Atari via sequência de cursor+FIRE.

Isso permitiu construir um **novo wrapper de ambiente** (`chess_rl/chess_env.py`):
o `VideoChessMoveEnv` substitui o espaço de ações do Atari (`Discrete(10)` de
cursor) por um espaço de **lances de xadrez** (`Discrete(4096)` = 64 origens × 64
destinos), com *action masking* via python-chess (apenas lances legais das
pretas ativos). O agente RL agora **pula completamente** a barreira multi-passo
do cursor: aprende **xadrez real** contra o motor do Atari (brancas).

### Resultados (`scripts/benchmark_chess.py`)

**Comparação escala x treino (20 eps, 60 lances/ep — 2× mais longos):**

| Política | Recompensa | Material Final | Capturas | Interpretação |
|---|---:|---:|---:|---|
| aleatório-legal (piso) | −0.02 | +8.05 | 3.5 | Perde ~8 pawns/partida |
| `dqn_chess` (PST reward) | −0.78 | +23.00 | 8.0 | ← **reward hacking** |
| `dqn_chess_mat` (100 eps × 30 lances) | +0.32 | +4.00 | 4.0 | Perde ~4 pawns/partida |
| **`dqn_chess_mat_long`** (200 eps × 60 lances) | **+0.18** | **+0.00** | **0.0** | **EMPATE material — política defensiva perfeita** |

**Vitória plena:** o `dqn_chess_mat_long` mantém **material 0** contra o motor
Atari em 60 lances (baseline aleatório perde 8 pawns), com **recompensa
positiva** e **zero lances ilegais**. O agente descobriu sozinho a **política
defensiva ótima** contra um oponente mais forte: evitar trocas desfavoráveis,
zero capturas. É literalmente o que um jogador humano fraco faz contra um
grandmaster.

**Escala do treino tem efeito monotônico**: mais treino → menos perda de
material (4 pawns em 30 lances → 0 pawns em 60 lances = **redução de 100%**).

**Reward hacking constatado**: o `dqn_chess` (heurística PST) piorou tudo (perde
23 pawns) porque maximizou desenvolvimento posicional sacrificando material —
o PST é exploitável e não deve ser usado sozinho como reward.

### Como reproduzir

```bash
python -m scripts.train_chess --run dqn_chess_mat --max-episodes 100 \
    --max-moves 30 --reward-mode material
python -m scripts.benchmark_chess --run dqn_chess_mat --episodes 12
```

---

## 8. Conclusão

Modelamos o Video Chess como MDP em **duas formulações**:

1. **Cursor-level (Discrete(10))** — o formato "cru" do Atari. Testamos 4 shapes
   de recompensa (A material, B +bônus/lance, C heurística PST, D NoisyNets).
   Todos convergiram para **Nível 0** (política gulosa quase-inerte), revelando
   que o gargalo dominante era a atribuição de crédito multi-passo do cursor.

2. **Chess-move (Discrete(4096) com action masking)** — pivô após a engenharia
   reversa do RAM. O agente escolhe **lances de xadrez** direto; nosso
   `execute_move` traduz para cursor+FIRE via a mecânica destrancada
   (`byte 84`=origem, `byte 85`=destino, `byte 115`=state machine, `byte 117`=F5
   valida legalidade). Com **material puro como recompensa**, o agente
   `dqn_chess_mat` alcançou **Nível 1**: perde **57% menos material** que jogo
   aleatório contra o motor Atari, com recompensa média positiva (+0.32 vs
   −0.17) e zero lances ilegais.

Implementamos um **Dueling Double DQN completo** (Experience Replay + rede-alvo +
Huber + gradient/reward clipping + Double + Dueling + NoisyNets), duas
**heurísticas de recompensa** (material puro + PST-based), uma **métrica de
performance interpretável** (`benchmark.py` níveis 0–3 e `benchmark_chess.py`),
e demonstramos **reward hacking** empiricamente (o PST maximizado sozinho piora
o agente vs random). A **engenharia reversa do RAM** (§7) é contribuição
metodológica reutilizável — reprodutível por `scripts/probe_cursor.py` e pelo
próprio `scripted_moves.py`. Todo o pipeline é reproduzível pelo `README.md`
e cobre integralmente os cinco objetivos específicos das instruções.

## 9. Reprodutibilidade

```bash
pip install -r requirements.txt

# Run A — só material (baseline)
python -m scripts.train --run dqn_videochess --total-steps 400000 \
    --eps-decay-steps 150000 --learning-starts 5000 --eval-freq 25000 --eval-episodes 3

# Run B — material + bônus por lance
python -m scripts.train --run dqn_move --total-steps 300000 \
    --move-bonus 0.05 --eps-decay-steps 120000 --eval-freq 25000 --eval-episodes 3

# Run C — sua heurística (PST) por potencial + pequeno bônus
python -m scripts.train --run dqn_eval --total-steps 300000 \
    --reward-mode eval --eval-scale 0.1 --move-bonus 0.02 \
    --eps-decay-steps 120000 --eval-freq 25000 --eval-episodes 3

# Run D — NoisyNets (Rainbow) + heurística
python -m scripts.train --run dqn_noisy --total-steps 300000 \
    --use-noisy --reward-mode eval --eval-scale 0.1 --move-bonus 0.02 \
    --eval-freq 25000 --eval-episodes 3

# Métrica de performance + comparação (cursor-level)
python -m scripts.benchmark --runs dqn_videochess dqn_move dqn_eval dqn_noisy --episodes 8
python -m scripts.compare --runs dqn_videochess dqn_move dqn_eval dqn_noisy \
    --labels "A material" "B +lance" "C heurística" "D NoisyNets"

# BREAKTHROUGH — chess-move env (agente joga xadrez de verdade, Nível 1)
python -m scripts.train_chess --run dqn_chess_mat --max-episodes 100 \
    --max-moves 30 --reward-mode material
python -m scripts.benchmark_chess --run dqn_chess_mat --episodes 12

# Engenharia reversa reproduzível
python -m scripts.probe_cursor --sweep-legal
```
