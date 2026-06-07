# Blank-tile reference crops

Example tiles for calibrating blank detection (see `6_blank_detection.ipynb`).

Save examples here with clear names, e.g.:
- `blank_u.png`   — a played blank (no point pip in the top-right)
- `letter_e.png`  — a normal lettered tile (with pip), for contrast

Once present, the notebook can measure each one's top-right dark fraction and
set `BLANK_THRESHOLD` from real data instead of a synthetic blank.
