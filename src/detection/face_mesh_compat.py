"""
Compatibility layer for MediaPipe face mesh.
Supports both legacy (mp.solutions.face_mesh) and MediaPipe 0.10+ (FaceLandmarker).
"""
import os
import sys
import time

def _get_face_landmarker_model_path():
    """Return path to face_landmarker.task. Downloads if missing."""
    # Prefer project models/ or env
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(base, "models")
    path = os.path.join(model_dir, "face_landmarker.task")
    if os.path.isfile(path):
        return path
    env_path = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    if env_path and os.path.isfile(env_path):
        return env_path
    # Download from MediaPipe hosted model
    try:
        import urllib.request
        os.makedirs(model_dir, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"Download face_landmarker model failed: {e}. Place face_landmarker.task in {model_dir} or set MEDIAPIPE_FACE_LANDMARKER_MODEL.", file=sys.stderr)
        raise


def create_face_mesh(config):
    """
    Returns a face mesh processor with .process(rgb_frame) -> result
    where result has .multi_face_landmarks (list of objects with .landmark[i].x, .y).
    """
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            # Legacy MediaPipe (< 0.10)
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return _LegacyFaceMesh(face_mesh)
    except Exception:
        pass

    # MediaPipe 0.10+ (tasks API)
    from mediapipe.tasks.python.core import base_options as base_options_lib
    from mediapipe.tasks.python.vision import face_landmarker as face_landmarker_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
    from mediapipe.tasks.python.vision.core import image as image_lib

    model_path = _get_face_landmarker_model_path()
    BaseOptions = base_options_lib.BaseOptions
    FaceLandmarker = face_landmarker_lib.FaceLandmarker
    FaceLandmarkerOptions = face_landmarker_lib.FaceLandmarkerOptions
    VisionRunningMode = running_mode_lib.VisionTaskRunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    return _TasksFaceMesh(landmarker, image_lib)


class _LegacyFaceMesh:
    """Wraps legacy mp.solutions.face_mesh.FaceMesh."""
    def __init__(self, face_mesh):
        self._face_mesh = face_mesh

    def process(self, rgb_frame):
        return self._face_mesh.process(rgb_frame)


class _FakeLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z


class _FakeFaceLandmarks:
    """Mimics legacy multi_face_landmarks[i] with .landmark[j].x, .y."""
    def __init__(self, landmarks_list):
        self.landmark = [_FakeLandmark(l.x, l.y, getattr(l, "z", 0)) for l in landmarks_list]


class _FakeResult:
    """Mimics legacy result with .multi_face_landmarks."""
    def __init__(self, face_landmarks_list):
        self.multi_face_landmarks = [_FakeFaceLandmarks(face) for face in face_landmarks_list]


class _TasksFaceMesh:
    """Wraps MediaPipe 0.10 FaceLandmarker and returns legacy-shaped results."""
    def __init__(self, landmarker, image_lib):
        self._landmarker = landmarker
        self._image_lib = image_lib
        self._timestamp_ms = 0

    def process(self, rgb_frame):
        self._timestamp_ms += 1
        ImageFormat = self._image_lib.ImageFormat
        mp_image = self._image_lib.Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.face_landmarks:
            return _FakeResult([])
        return _FakeResult(result.face_landmarks)
