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
    conflicting_subject: Optional[str] = Field(default=None, description="Name of conflicting subject if biometric collision detected")


class FacePreviewRequest(BaseSchema):
    images_base64: List[str] = Field(default_factory=list, alias="images", description="Array of base64 encoded images to preview & curate")


class FacePreviewItem(BaseSchema):
    crop_id: str
    image_b64: str
    sharpness_score: float
    confidence: float
    is_recommended: bool
    status_label: str = "Optimal Quality"


class FacePreviewResponse(BaseSchema):
    status: str = Field(default="success")
    previews: List[FacePreviewItem] = Field(default_factory=list)
    total_detected: int = 0


class FaceCommitEnrollmentRequest(BaseSchema):
    full_name: str = Field(..., min_length=1, alias="name", description="Full name of subject to enroll")
    phone_number: Optional[str] = Field(default="", description="Contact phone number")
    occupation: Optional[str] = Field(default="", description="Occupation or role title")
    dob: Optional[str] = Field(default="", description="Date of birth")
    clearance: Optional[str] = Field(default="LEVEL 1", description="Access clearance level")
    address: Optional[str] = Field(default="", description="Primary address")
    threat_level: Optional[str] = Field(default="LOW", description="Assigned threat level assessment")
    approved_crops_base64: List[str] = Field(default_factory=list, alias="approved_crops", description="List of approved face crop base64 strings to index")


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


class AttendanceConfigRequest(BaseSchema):
    class_start_time: Optional[str] = Field(default=None, example="08:30", description="Class start time in HH:MM or HH:MM:SS format")
    class_end_time: Optional[str] = Field(default=None, example="11:30", description="Class end time in HH:MM or HH:MM:SS format")
    cutoff_time: Optional[str] = Field(default=None, example="08:30", description="Backward compatible alias for start time")
    grace_period_minutes: Optional[int] = Field(default=0, description="Grace period in minutes")


class AttendanceConfigResponse(BaseSchema):
    status: str = Field(default="success")
    class_start_time: Optional[str] = "08:30:00"
    class_end_time: Optional[str] = "11:30:00"
    cutoff_time: str
    grace_period_minutes: int = 0
    current_time: Optional[str] = None
    today: Optional[str] = None
    message: Optional[str] = None


class AttendanceChecklistItem(BaseSchema):
    person_name: str
    status: str = Field(..., example="PRESENT", description="PRESENT, LATE, ABSENT, PENDING, or MANUAL_PRESENT")
    check_in_time: Optional[str] = None
    track_id: Optional[int] = None
    occupation: Optional[str] = "STUDENT"
    clearance: Optional[str] = "REGISTERED"
    snapshot_path: Optional[str] = ""
    is_manual: Optional[int] = 0
    override_reason: Optional[str] = ""


class AttendanceChecklistResponse(BaseSchema):
    status: str = Field(default="success")
    date: str
    class_start_time: Optional[str] = "08:30:00"
    class_end_time: Optional[str] = "11:30:00"
    cutoff_time: str
    grace_period_minutes: int = 0
    total_enrolled: int
    present_count: int
    late_count: int
    absent_count: int
    pending_count: Optional[int] = 0
    excused_count: Optional[int] = 0
    attended_count: int
    attendance_rate: Optional[float] = 0.0
    checklist: List[AttendanceChecklistItem] = Field(default_factory=list)
    course: Optional[Dict[str, Any]] = None


class AttendanceManualMarkRequest(BaseSchema):
    person_name: str
    status: str = Field(..., example="PRESENT", description="PRESENT, LATE, ABSENT, or MANUAL_PRESENT")
    date: Optional[str] = None
    custom_time: Optional[str] = None
    reason: Optional[str] = ""
    course_id: Optional[int] = None


class AttendanceCheckAllRequest(BaseSchema):
    status: Optional[str] = Field(default="PRESENT", example="PRESENT", description="PRESENT, LATE, or ABSENT")
    date: Optional[str] = None
    course_id: Optional[int] = None


class UnusualPersonAlertItem(BaseSchema):
    track_id: int
    first_seen: str
    last_seen: str
    snapshot_path: Optional[str] = ""
    snapshot_url: Optional[str] = ""
    status: str = "ACTIVE"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UnusualPersonListResponse(BaseSchema):
    status: str = Field(default="success")
    alerts: List[UnusualPersonAlertItem] = Field(default_factory=list)
    total_alerts: int = 0


class UnusualPersonDismissRequest(BaseSchema):
    track_id: int


class UnusualPersonEnrollRequest(BaseSchema):
    track_id: int
    person_name: str


class CourseCreateRequest(BaseSchema):
    code: str
    name: str
    department: Optional[str] = "GENERAL"


class CourseEnrollStudentRequest(BaseSchema):
    person_name: str


class ScheduleCreateRequest(BaseSchema):
    day_of_week: int
    start_time: str
    end_time: str
    grace_minutes: Optional[int] = 15



