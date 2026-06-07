# WordfeudRL — Project Summary

Goal: play **Wordfeud** (Danish Scrabble variant) from a phone screenshot. The
end target is a reinforcement-learning agent, so the code is built around the
RL primitives: read the **state** from an image, enumerate the legal **actions**
(moves), score them, and produce the **next state**.

## Pipeline

```
screenshot ──▶ WordfeudMap ──▶ (letters grid, bonus grid) ──▶ WordfeudEngine ──▶ scored moves
  .jpeg          vision           split_board()                 generator        + next states
```

1. **Vision** ([src/screenshot_to_map.py](src/screenshot_to_map.py)) turns a
   screenshot into a 15×15 board (bonus squares + placed letters) and reads the
   7-tile rack.
2. **Dictionary** ([src/dictionary.py](src/dictionary.py)) loads the Danish word
   list, grouped by length.
3. **Engine** ([src/move_generator.py](src/move_generator.py)) generates, scores,
   and applies every legal move.

Notebooks `1_`, `2_`, `3_` are the interactive driver for each stage.

## Files

| File | Role |
|------|------|
| [src/screenshot_to_map.py](src/screenshot_to_map.py) | `WordfeudMap`: image → board map, tile letter recognition, rack reading |
| [src/dictionary.py](src/dictionary.py) | `load_dictionary()`: cleans `all_words.csv` → `{length: [words]}` |
| [src/move_generator.py](src/move_generator.py) | `WordfeudEngine`, `Move`, `split_board`, `plot_board` |
| [1_screenshot_to_map.ipynb](1_screenshot_to_map.ipynb) | Vision development |
| [2_dictionary.ipynb](2_dictionary.ipynb) | Dictionary building |
| [3_interaction.ipynb](3_interaction.ipynb) | Full loop: read board → generate moves → visualise |
| `all_words.csv` | DDO Danish full-form word list (source data) |
| `letters/` | Per-letter template images for tile recognition |
| `imgs/` | Input screenshots |

## Vision — `WordfeudMap`

- **Board colour map** (`img_to_map`): samples each tile centre, snaps the RGB to
  the nearest of 6 known board colours → a 15×15 grid of integer codes.
- **Letter recognition** (`tile_to_text`): binarises a tile, strips the border,
  drops border-connected blobs, then scale+translate template-matches each
  letter image with a chamfer-distance score. Best score wins.
- **Rack** (`available_letters`): same recognition on the 7 tiles at the bottom.
- **`map_with_letters()`**: combined grid — recognised letters on normal tiles,
  bonus code (as string) elsewhere. Use `split_board()` to split it back apart.

### Coordinate convention
All grids are `[row][col]`, row 0 = top, col 0 = left. The board map and tile
extraction both use this frame consistently.
> Fixed bug: `map_with_letters` previously read the bonus code from the
> transposed cell (`map[yi, xi]`), misaligning bonus squares from letters. Now
> reads `[xi, yi]`.

## Bonus-square codes

Set by the board colour map; meaning + multipliers live in `move_generator`
(`BONUS_MULT`) and can be edited there if a code is mislabelled.

| Code | Colour | Square | Effect |
|------|--------|--------|--------|
| 0 | dark | board gap | — |
| 1 | light | normal tile | — |
| 2 | green | triple letter | letter ×3 |
| 3 | blue | double letter | letter ×2 |
| 4 | orange | double word | word ×2 |
| 5 | red | triple word | word ×3 |

## Engine — `WordfeudEngine`

Classic Scrabble move generator: a **trie** of the dictionary + **anchor**
squares + **cross-checks**, run once horizontally and once on the transposed
board for vertical plays. Builds from 496k words in ~1s; generates a full move
list in ~0.04s.

```python
from src.move_generator import WordfeudEngine, split_board, plot_board

words  = [w for lst in dictionary.values() for w in lst]
engine = WordfeudEngine(words)

letters, bonus = split_board(mapped)             # clean grids from the vision output
moves = engine.legal_moves(letters, bonus, my_letters)   # actions, best score first

best = moves[0]
next_letters = engine.next_state(letters, best)  # next state
```

### `Move`
`new_tiles` = the action: `[(row, col, letter, is_blank), ...]`. Plus `score`,
and for display `word` / `direction` (`'H'`/`'V'`) / `start`.

### API
| Method | Purpose |
|--------|---------|
| `legal_moves(letters, bonus, rack)` | all legal moves, deduped, sorted by score |
| `next_state(letters, move)` | board grid after applying a move |
| `score_move(letters, bonus, new_tiles)` | score an arbitrary tile placement |

### Scoring rules
- Letter/word multipliers apply **only to newly placed tiles**.
- **All** formed words count (the main word + every cross-word).
- **Blanks** (`'*'` in the rack) score 0 points but can be any letter.
- **+40** bonus for using all 7 rack tiles (Wordfeud bingo; Scrabble uses 50).

### Inputs
- `letters`: 15×15, lowercase, `''` = empty.
- `bonus`: 15×15 int codes (table above).
- `rack`: list of letters; `'*'` = blank.

## Visualisation — `plot_board`

`plot_board(letters, bonus, move=None, ax=None, title=None)` draws the board with
coloured bonus squares and placed tiles. Passing a `move` highlights its new
tiles in gold and dims the rest — used in [3_interaction.ipynb](3_interaction.ipynb)
to show *current → proposed move → next state* and a top-candidates gallery.

## RL stack

The training loop never touches vision — it runs a pure simulator.

```
WordfeudGame (sim) ──▶ WordfeudEnv (gym) ──▶ agent picks a candidate ──▶ reward
  bag/turns/rules        candidate-eval         (eval.py / rl.py)        score+margin
```

| File | Role |
|------|------|
| [src/game.py](src/game.py) | `WordfeudGame`: bag, two racks, turns, scoring, end-game rack adjustment |
| [src/agents.py](src/agents.py) | `GreedyAgent` (the opponent/baseline), `RandomAgent` |
| [src/env.py](src/env.py) | `WordfeudEnv`: Gymnasium single-agent env vs greedy |
| [src/eval.py](src/eval.py) | `evaluate()` — win rate / margin over N games; `best_candidate_policy` |
| [src/rl.py](src/rl.py) | `LinearCandidateAgent` + batched REINFORCE `train()` (numpy starter) |
| [src/rl_torch.py](src/rl_torch.py) | PyTorch actor-critic candidate agent + `train()` |
| [src/boards.py](src/boards.py) | `STANDARD_BOARD`, `random_board()` (Wordfeud random-mode layouts) |
| [src/probability.py](src/probability.py) | Unseen-tile pool + word/letter draw probabilities |
| [src/lookahead.py](src/lookahead.py) | `LookaheadAgent` (2-ply Monte-Carlo) + `duel()` |
| [4_environment.ipynb](4_environment.ipynb) | Demo: step, render, baseline eval, train (+ save model) |
| [5_play_from_screenshot.ipynb](5_play_from_screenshot.ipynb) | Read a screenshot → pick the best move with the learned (or greedy) policy |

`env.choose_action(engine, letters, bonus, rack, policy)` ranks the legal moves
for a single position using the same candidate features as training, so any
`policy(obs, info)` (greedy or learned) works at deployment. The learned agent
persists via `LinearCandidateAgent.save/load` (`models/linear_agent.npy`); the
screenshot notebook loads it if present, else falls back to greedy.

### Design decisions
- **Action interface = candidate evaluation.** Each turn the generator emits the
  legal moves; the env exposes the **top-K** as a feature matrix + action mask,
  and `action` indexes into them (last index = pass/swap). Full `Move` objects
  are in `info["moves"]` for a custom policy. Fits maskable policies; avoids an
  intractable fixed action space.
- **Single-agent vs greedy.** The greedy opponent plays its turn *inside*
  `step`, so the agent sees a clean one-player MDP. Self-play is a later upgrade.
- **Reward (`reward_mode`).** `"margin"` (default) rewards only the terminal
  score margin — the winning signal an agent must learn to beat greedy.
  `"shaped"` adds dense per-move score: easy to learn but pulls toward greedy
  (own-score maximisation). See "Beating greedy" below.

### Game rules modelled
105-tile Danish bag (103 letters + 2 blanks), draw/refill, pass, swap (bag ≥
rack), end when a player empties the rack with an empty bag (finisher gains
opponents' leftover points) or after `max_scoreless` consecutive non-scoring
turns (each loses own leftovers).

### Candidate features (per move)
`score, n_tiles, is_vertical, start_row, start_col, word_len, leave_points`
(`leave_points` = point-sum of tiles kept on the rack — the classic "leave"
signal greedy ignores).

### Status / results
The numpy REINFORCE trainer is a **correctness scaffold**, not a strong player:
the loop is stable (mean return rises, score weight learns positive) but the
linear policy lands around greedy (~43–50% win), since greedy already maximises
immediate score. **To actually beat greedy:** replace the linear scorer with a
neural net over the board/rack planes (the env already provides them), add
proper leave/positional features, and train longer — then add self-play.

## Beating greedy: two paths

Greedy is per-move score-optimal, so beating it means valuing what it ignores
(rack leave, board danger). Two approaches are implemented:

### A. Monte-Carlo lookahead (recommended) — `lookahead.py`
No training. For each candidate move, simulate over racks sampled from the
**unseen-tile pool** ([probability.py](src/probability.py)):

```
2-ply:  value(m) = m.score − E[ opponent's best reply on board+m ] + leave
3-ply:  value(m) = m.score − E[ opponent reply ] + E[ my best next turn ] + leave
```

This captures **defense** automatically — a move that opens a triple-word scores
well now but raises the opponent's expected reply, so its value drops. 2-ply
beat greedy head-to-head on matched random boards (+~10 pp win rate, +~8 margin
at only 6 samples / 8 candidates).

Tuning levers (accuracy vs speed): `n_samples`, `max_candidates`, and `plies`
(2 or 3). **3-ply** also simulates *your* next turn, so it values the rack leave
directly (a good leave → stronger follow-up); ~2× the cost of 2-ply. Cost is
`K·N` (×2 at 3-ply) move-generations per turn — an inference-time search, too
slow as a self-play opponent at scale.

**Exact endgame:** when the bag is empty the unseen pool *is* the opponent's
rack, so the agent drops sampling and evaluates a single deterministic rollout
(verified identical across RNG seeds), plus the go-out bonus (you gain the
opponent's leftover tile points when you empty your rack first).

The **information model**: you know the full bag, the board, and your own rack;
everything else is the unseen pool (opponent rack + remaining bag). `probability.
py` exposes it and answers "probability a draw can form word W" (Monte-Carlo,
blank-aware) and expected letter counts (exact, by linearity).

### B. Model-free RL — `rl.py` (linear), `rl_torch.py` (actor-critic)
The Gymnasium env + candidate-evaluation policy. The torch actor-critic trains
stably (decoupled critic, reward scaling) but, with the **shaped** reward, chases
own-score and only matches greedy; use `reward_mode="margin"` (now the env
default) to train on the winning signal, plus richer/defensive features and far
more episodes. Slower path to surpassing greedy than (A), but it's the route to a
fast learned policy and eventual self-play.

## Blank tiles (end-to-end)

Blanks are tracked everywhere they affect play:
- **Rack blank** = `'*'`; the generator can play it as any letter, scoring 0 for
  that tile (`is_blank` flag on each `Move` tile).
- **Board blank** = a `(row, col)` in a `board_blanks` set, threaded through
  `legal_moves` / `score_move` / `_score` so words built across it score the
  blank cell as 0. The simulator updates `WordfeudGame.board_blanks` as blanks
  are played; `lookahead` carries it (and adds blanks each ply); `probability.
  unseen_tiles` debits a `'*'` (not the letter) for known board blanks.
- **Vision** ([screenshot_to_map.py](src/screenshot_to_map.py)): `find_pip` /
  `is_blank` detect a tile with no point pip (small non-border dark blob in the
  extreme top-right). `read_state()` returns `(letters, bonus, board_blanks,
  rack)` — blank-aware — and `available_letters()` emits `'*'`. See
  [6_blank_detection.ipynb](6_blank_detection.ipynb) for the calibration.

## Known limitations / TODO

- **No recognition confidence threshold.** `tile_to_text` always returns its
  best guess, so *empty* normal tiles get a spurious letter. Add a minimum score
  cutoff that returns `None` for empty tiles before trusting boards from real
  screenshots.
- **`split_board` drops the bonus code under existing letters** (sets it to 1).
  Harmless for scoring future moves (bonuses only matter on empty target cells),
  but pass `WF.map` directly as `bonus` if you want the true value everywhere.
- **Strong learned agent not built yet** — the env + a working REINFORCE
  scaffold exist; next is a neural value/policy net over the board+rack planes,
  richer leave/positional features, longer training, then self-play.
- **Generator perf** — `legal_moves` rebuilds cross-checks + walks a Python trie
  each call (~0.04 s/turn; a full game ~0.5 s). Fine now; likely the bottleneck
  for large-scale self-play. Optimise (incremental cross-checks / numpy / Cython)
  if training throughput bites.
- **Bonus board in the env** is a flat placeholder (`bonus[7][7]=4`); plug in a
  real Wordfeud layout (`WordfeudMap(path).map.tolist()`) for realistic play.
