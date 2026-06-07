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

## Known limitations / TODO

- **No recognition confidence threshold.** `tile_to_text` always returns its
  best guess, so *empty* normal tiles get a spurious letter. Add a minimum score
  cutoff that returns `None` for empty tiles before trusting boards from real
  screenshots.
- **`split_board` drops the bonus code under existing letters** (sets it to 1).
  Harmless for scoring future moves (bonuses only matter on empty target cells),
  but pass `WF.map` directly as `bonus` if you want the true value everywhere.
- **RL agent not built yet** — the engine provides `(action, next_state)`; the
  policy/value learning is the next stage.
