"""
Standardized entrypoint alias delegating to src.api.cctv_server.
Guarantees thread-safe single camera stream handle execution across the process.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.cctv_server import app, start_server, camera_stream

if __name__ == "__main__":
    start_server()
