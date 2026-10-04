import os
import sys
import json
import sqlite3
import asyncio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "faces.db")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

from src.database.audit_repository import AuditRepository
audit_repo = AuditRepository(DB_PATH)

from src.inference.engine import BiometricTrackingEngine
import src.inference.live_cctv as live_cctv

engine: BiometricTrackingEngine = live_cctv.engine

import importlib

FastMCP = None
try:
    _fastmcp_mod = importlib.import_module("fastmcp")
    FastMCP = getattr(_fastmcp_mod, "FastMCP", None)
except ImportError:
    pass

if FastMCP is not None:
    mcp = FastMCP("Enterprise macOS CCTV Fleet Intelligence")
else:
    # Lightweight fallback FastMCP wrapper if fastmcp package is not installed (e.g. Python < 3.10)
    class DummyMCP:
        def __init__(self, name):
            self.name = name
            self.tools = {}
        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = func
                return func
            return decorator
    mcp = DummyMCP("Enterprise macOS CCTV Fleet Intelligence")

@mcp.tool()
async def get_active_tracks() -> str:
    """
    Retrieves real-time active track IDs, bounding boxes, and biometric identity matches 
    currently being monitored by the CCTV pipeline.
    """
    active_subjects = engine.get_active_tracks()
    active_list = [
        {"track_id": s.track_id, "person_name": s.name} 
        for s in active_subjects
    ]
    return json.dumps({
        "status": "success",
        "total_active_tracks": len(active_list),
        "active_tracks": active_list
    }, indent=2)

@mcp.tool()
async def get_audit_events(limit: int = 50) -> str:
    """
    Queries the SQLite security audit log database (faces.db WAL mode) to retrieve 
    recent biometric identification events.
    """
    events = await audit_repo.get_recent_events(limit=limit)

    return json.dumps({
        "status": "success",
        "total_events": len(events),
        "events": events
    }, indent=2)

@mcp.tool()
async def search_faiss_identity(person_name: str) -> str:
    """
    Searches enrolled biometric subject profiles and metadata map for an individual by name.
    """
    matches = []
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                for idx_str, entry in meta.items():
                    name = entry.get("name", str(entry)) if isinstance(entry, dict) else str(entry)
                    if person_name.lower() in name.lower():
                        matches.append({"profile_id": int(idx_str) if str(idx_str).isdigit() else idx_str, "person_name": name})
            elif isinstance(meta, list):
                for idx, entry in enumerate(meta):
                    name = entry.get("name", str(entry)) if isinstance(entry, dict) else str(entry)
                    if person_name.lower() in name.lower():
                        matches.append({"profile_id": idx, "person_name": name})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    return json.dumps({
        "status": "success",
        "query": person_name,
        "matches_found": len(matches),
        "profiles": matches
    }, indent=2)

if __name__ == "__main__":
    if hasattr(mcp, "run"):
        mcp.run(transport="stdio")
    else:
        print("[FastMCP Server] MCP tools registered successfully:")
        print(list(mcp.tools.keys()))
