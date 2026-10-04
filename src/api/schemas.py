from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class FaceEnrollmentRequest(BaseSchema):
    full_name: str = Field(..., min_length=1, description="Full name of subject to enroll")
    phone_number: Optional[str] = Field(default="", description="Contact phone number")
    occupation: Optional[str] = Field(default="", description="Occupation or role title")
    dob: Optional[str] = Field(default="", description="Date of birth")
    clearance: Optional[str] = Field(default="LEVEL 1", description="Access clearance level")
    address: Optional[str] = Field(default="", description="Primary address")
    threat_level: Optional[str] = Field(default="LOW", description="Assigned threat level assessment")
    image_base64: Optional[str] = Field(default="", description="Single base64 encoded face image")
    images_base64: Optional[List[str]] = Field(default_factory=list, description="Array of base64 encoded pose face images")


class FaceEnrollmentResponse(BaseSchema):
    status: str = Field(..., example="success")
    message: str = Field(...)
    profile_id: Optional[int] = Field(default=None)
    total_profiles: Optional[int] = Field(default=None)


class CameraFeedConfig(BaseSchema):
    stream_id: str = Field(..., example="cam_0")
    source: Any = Field(..., description="RTSP URL, video file path, or device index integer")
    width: int = Field(default=854)
    height: int = Field(default=480)


class CameraStatusResponse(BaseSchema):
    status: str = Field(..., example="success")
    active_streams: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class ActiveTrackSchema(BaseSchema):
    track_id: int
    name: str
    match_distance: float = Field(default=0.45)
    meta: Dict[str, Any] = Field(default_factory=dict)
    keypoints: Optional[List[Dict[str, float]]] = Field(default=None, description="17 pose keypoints if PoseOnlyTracking is active")


class AuditEventSchema(BaseSchema):
    id: int
    timestamp: str
    person_name: str
    track_id: int
    match_distance: float
    snapshot_path: Optional[str] = ""
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TelemetryResponse(BaseSchema):
    status: str = Field(..., example="success")
    events: List[AuditEventSchema] = Field(default_factory=list)
    active_tracks: List[ActiveTrackSchema] = Field(default_factory=list)
    total_profiles: int = Field(default=0)
    fps: float = Field(default=0.0)
    message: Optional[str] = None


class SystemModeRequest(BaseSchema):
    mode: str = Field(..., example="SURVEILLANCE", description="SURVEILLANCE or ENROLLMENT")
    pose_only_mode: Optional[bool] = Field(default=None, description="Toggles 17-keypoint PoseOnlyTracking mode")


class SystemModeResponse(BaseSchema):
    status: str = Field(..., example="success")
    mode: str
    pose_only_mode: bool = False
    message: Optional[str] = None


class HealthResponse(BaseSchema):
    status: str = Field(default="ok", example="ok")
    timestamp: float


class ReadinessResponse(BaseSchema):
    status: str = Field(..., example="ready")
    faiss_loaded: bool
    cameras_running: bool
    total_profiles: int
    reason: Optional[str] = None
