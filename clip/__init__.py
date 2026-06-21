# Re-export the public API so `import clip` exposes `load`, `tokenize`, and
# `available_models` (matches the upstream OpenAI CLIP package layout).
# Without this file `clip/` was only an implicit namespace package, so
# `import clip; clip.load(...)` failed and the absolute imports inside
# clip/clip.py (e.g. `from clip.model import build_model`) were fragile.
from .clip import *  # noqa: F401,F403
