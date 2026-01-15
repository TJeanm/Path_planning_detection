# Path Planning (Cone Detection + Local RRT*)

A small **Python + Pygame** project that simulates a race track defined by **cones** (blue/yellow) and makes a car plan a **local trajectory in real time** using a simple **perception model** (field of view + range + short-term memory) and a **local RRT\*** planner.

The goal is to visualize how “what the car sees” impacts the planned path.

---

## Features

- Track loaded from CSV cone maps (blue/yellow + optional `car_start`)
- Simple perception:
  - limited **FOV (~100°)**  
  - limited **view distance (~10 m)**
  - a small grid-based **memory** of seen cones
- Real-time **local planning** with **RRT\*** (replanned every frame)
- Basic path following with a **pure pursuit**-style target point
- Visualization in Pygame:
  - cones
  - detected/visible cones
  - RRT tree
  - planned path
  - camera zoom with the mouse wheel

---

## Requirements

- Python 3.9+ (works best with 3.10+)
- Packages:
  - `numpy`
  - `scipy`
  - `pygame`

---

## Installation

From the repository root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -U pip
pip install numpy scipy pygame
```

> On some Linux setups, installing `pygame` may require SDL dependencies (distribution-specific).

---

## Run

From the repository root:

```bash
python src/main.py
```

Then pick one of the built-in tracks:
- `small_track`
- `hairpins_increasing_difficulty`
- `peanut`

The simulation starts automatically.

---

## Controls

- **Mouse wheel**: zoom in / zoom out  
(Everything else is automatic: planning + driving)

---

## Track / CSV format

Track files are stored in `data/` and are read with `csv.DictReader`.

Minimum required columns:

- `tag` (e.g., `blue`, `yellow`, `car_start`)
- `x`
- `y`

Extra columns (like `direction`, variances, etc.) are allowed and simply ignored by the loader.

Example:

```csv
tag,x,y
blue,1.0,2.0
yellow,1.5,2.2
car_start,0.0,0.0
```

To add your own track:
1. Drop your CSV into `data/`
2. Add it in `TRACKS` inside `src/main.py`

---

## Project structure

```
data/                 # cone maps (CSV)
src/
  main.py             # track selector + launcher
  core/
    process_path_local.py   # perception + Local RRT* planner
  ui/
    process_pygame.py       # Pygame simulation + rendering
    camera.py               # camera + zoom
  utils/
    track_utils.py          # CSV loading + world bounds
```

---

## Notes / Troubleshooting

- If `ModuleNotFoundError: utils` happens, make sure you run **from the repo root** with:
  `python src/main.py`
- If the planner fails to find a path in a frame, it falls back to a simple straight segment to the local target.

---

## License

No license file is included in this repository (default: all rights reserved).  
If you plan to publish it, consider adding an explicit license (MIT, Apache-2.0, etc.).
