import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.api.cctv_server import start_server, app

if __name__ == "__main__":
    start_server()
