import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.tune_tracker import run_bytetrack_grid_search

if __name__ == "__main__":
    run_bytetrack_grid_search()
