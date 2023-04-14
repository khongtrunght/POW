import os
from pathlib import Path

# this file is DATN/src/utils/paths.py
# PRỌEECTPATH is DATN
PROJECT_PATH = Path(__file__).parent.parent.parent
# PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(PROJECT_PATH, "weights")
COIN_PATH = os.path.join(PROJECT_PATH, "Datasets/COIN")
YC_PATH, CT_PATH = None, None
