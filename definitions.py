import os
from pathlib import Path
import json


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(Path(ROOT_DIR) / "data" / "config.json") as f:
    SETTINGS = json.load(f)["settings"]