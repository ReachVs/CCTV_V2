import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.database.db_setup import enroll_faces

if __name__ == "__main__":
    enroll_faces()
