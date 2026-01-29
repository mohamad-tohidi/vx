from typing import Final

# inventory/embedders.py
TEMPLATE_EMBEDDERS: Final[str] = """\
from typing import List

def bge_m3(text: str) -> List[float]:
    # Placeholder: In reality, load model and inference
    return [0.1, 0.2, 0.3]

def openai_small(text: str) -> List[float]:
    # Placeholder: API call logic
    return [0.9, 0.1, 0.5]
"""

# inventory/chunkers.py
TEMPLATE_CHUNKERS: Final[str] = """\
from typing import List

def simple_split(text: str, chunk_size: int = 512) -> List[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
"""

# experiments/exp_01_baseline.py
TEMPLATE_EXP_BASELINE: Final[str] = """\
from embed_lab import ExperimentConfig
from inventory.embedders import bge_m3
from inventory.chunkers import simple_split

config = ExperimentConfig(
    name="exp_01_baseline",
    
    # Ingredients
    embedder=bge_m3,
    chunker=simple_split,
    
    # Hyperparameters
    vector_size=1024,
    batch_size=32,
    learning_rate=2e-5
)
"""

# main.py (The project entrypoint)
TEMPLATE_MAIN: Final[str] = """\
import sys

def main() -> None:
    print("Run experiments using: uv run embed-lab train experiments/exp_01_baseline.py")

if __name__ == "__main__":
    main()
"""

TEMPLATE_GITIGNORE: Final[str] = """\
__pycache__/
*.py[cod]
.venv/
results/
.env
"""
