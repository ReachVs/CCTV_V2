"""
Security & Hardening Module for AI CCTV Surveillance Engine.
============================================================
Provides unified authentication, role-based authorization, rate limiting,
secure file delivery, WebSocket origin protection, and PII data masking.
Complies with OWASP Top 10 A01 (Access Control) & A04 (Insecure Design).
"""
import os
import time
import hmac
import threading
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from collections import defaultdict

from fastapi import Request, WebSocket, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse


# =====================================================================
# Configuration & Host Management
# =====================================================================

def get_server_host() -> str:
    """
    Returns the server bind host.
    Defaults to 127.0.0.1 (secure loopback) unless explicitly configured.
    """
    return os.environ.get("CCTV_HOST", os.environ.get("HOST", "127.0.0.1"))


def get_configured_api_key() -> Optional[str]:
    """Retrieves configured administrative API key from environment."""
    key = os.environ.get("CCTV_API_KEY", os.environ.get("CCTV_ADMIN_TOKEN", "")).strip()
    return key if key else None


# =====================================================================
# Authentication & Authorization (RBAC)
# =====================================================================

def is_admin_request(request: Request) -> bool:
    """
    Evaluates whether the incoming HTTP request holds verified admin credentials.
    Supports 'X-API-Key' header, 'Authorization: Bearer <key>', and 'cctv_token' cookie.
    In local development mode without a configured key, allows loopback clients.
    """
    configured_key = get_configured_api_key()
    
    # Extract client token
    token = request.headers.get("X-API-Key")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
    if not token:
        token = request.cookies.get("cctv_token")
    if not token:
        token = request.query_params.get("api_key")

    if configured_key:
        return bool(token and hmac.compare_digest(token.strip(), configured_key))

    # Dev mode fallback: If no API key is configured and CCTV_ENV is not production,
    # allow local loopback clients (127.0.0.1, ::1, testclient)
    is_prod = os.environ.get("CCTV_ENV", "").lower() == "production"
    if not is_prod:
        client_host = request.client.host if request.client else "testclient"
        if client_host in ("127.0.0.1", "::1", "testclient", "localhost"):
            return True

    return False


async def require_admin_auth(request: Request) -> None:
    """
    FastAPI dependency enforcing administrative authorization.
    Rejects unauthorized mutation or sensitive access with 401/403.
    """
    if not is_admin_request(request):
        configured_key = get_configured_api_key()
        if configured_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Administrative access requires a valid X-API-Key or Bearer token.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrative action forbidden. CCTV_API_KEY is not configured for non-local access."
            )


# =====================================================================
# In-Memory Rate Limiter (DoS Protection)
# =====================================================================

class InMemoryRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    Limits requests per client IP without external infrastructure dependencies.
    """
    def __init__(self, max_requests: int = 15, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self._lock:
            cutoff = now - self.window_seconds
            timestamps = [t for t in self._history[client_ip] if t > cutoff]
            if len(timestamps) >= self.max_requests:
                self._history[client_ip] = timestamps
                return False
            timestamps.append(now)
            self._history[client_ip] = timestamps
            return True

    def check(self, request: Request) -> None:
        """FastAPI dependency-compatible check method."""
        client_ip = request.client.host if request.client else "127.0.0.1"
        if not self.is_allowed(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.max_requests} requests per {int(self.window_seconds)}s."
            )


# Global instances for resource-intensive operations
enrollment_rate_limiter = InMemoryRateLimiter(max_requests=10, window_seconds=60.0)


# =====================================================================
# WebSocket Security & Origin Validation
# =====================================================================

def is_allowed_websocket_origin(websocket: WebSocket) -> bool:
    """
    Validates WebSocket Origin header against expected host to prevent
    Cross-Site WebSocket Hijacking (CSWSH).
    """
    origin = websocket.headers.get("origin")
    if not origin:
        # Standard local non-browser clients (OpenCV, python-socketio) omit origin
        return True

    host = websocket.headers.get("host")
    parsed = urlparse(origin)
    origin_host = parsed.netloc or parsed.path
    if not origin_host:
        return True

    clean_origin = origin_host.split(":")[0].lower()
    clean_host = (host.split(":")[0] if host else "").lower()

    if clean_origin in ("localhost", "127.0.0.1", "::1"):
        return True

    if clean_host and clean_origin == clean_host:
        return True

    allowed_env = os.environ.get("CCTV_ALLOWED_ORIGINS", "")
    if allowed_env:
        allowed = [a.strip().lower() for a in allowed_env.split(",") if a.strip()]
        if clean_origin in allowed:
            return True

    return False


# =====================================================================
# Controlled Biometric File Delivery (Guards Directory Traversal)
# =====================================================================

SAFE_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

def safe_file_response(base_dir: str, filename: str) -> FileResponse:
    """
    Safely delivers an image file from base_dir while strictly preventing
    path traversal attacks, directory enumeration, and execution of arbitrary files.
    """
    if not filename or not isinstance(filename, str):
        raise HTTPException(status_code=400, detail="Invalid filename parameter.")

    # Guard against traversal characters
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Path traversal characters are prohibited.")

    clean_name = os.path.basename(filename.strip())
    if not clean_name.lower().endswith(SAFE_IMAGE_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Requested file type is not an authorized image format.")

    abs_base = os.path.abspath(base_dir)
    target_path = os.path.abspath(os.path.join(abs_base, clean_name))

    # Strict containment check
    if os.path.commonpath([abs_base, target_path]) != abs_base or not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="File not found.")

    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Content-Type-Options": "nosniff"
    }
    return FileResponse(target_path, headers=headers)


# =====================================================================
# PII Redaction & Data Sanitization
# =====================================================================

def sanitize_subject_dossier(meta: Dict[str, Any], is_admin: bool = False) -> Dict[str, Any]:
    """
    Redacts sensitive personally identifiable information (PII) from subject profiles
    unless the caller possesses verified administrative clearance.
    """
    if is_admin:
        return dict(meta)

    sanitized = dict(meta)
    
    # Mask phone number (keep last 4 digits if present)
    phone = str(sanitized.get("phone", "N/A"))
    if phone and phone != "N/A" and len(phone) >= 7:
        sanitized["phone"] = f"***-***-{phone[-4:]}"
    else:
        sanitized["phone"] = "REDACTED"

    # Mask residential address
    if "address" in sanitized and sanitized["address"] != "SYSTEM DATABASE":
        sanitized["address"] = "REDACTED (RESTRICTED)"

    # Mask Date of Birth
    if "dob" in sanitized and sanitized["dob"] != "N/A":
        sanitized["dob"] = "REDACTED (RESTRICTED)"

    # Mask Threat Level and Clearance for non-admin viewers
    if "clearance" in sanitized:
        sanitized["clearance"] = "RESTRICTED (AUTHENTICATION REQUIRED)"
    if "threat_level" in sanitized:
        sanitized["threat_level"] = "CONFIDENTIAL"

    return sanitized
