"""Sub-module containing the camera configuration."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from metasim.utils.configclass import configclass


@configclass
class BaseCameraCfg:
    """Base camera configuration."""

    name: str = "camera0"
    """Name of the camera. Defaults to "camera0". Different cameras should have different names, so if you add multiple cameras, make sure to give them unique names."""
    data_types: list[Literal["rgb", "depth", "instance_seg", "instance_id_seg"]] = ["rgb", "depth"]
    """List of sensor types to enable for the camera. Defaults to ["rgb", "depth"]."""
    width: int = 256
    """Width of the image in pixels. Defaults to 256."""
    height: int = 256
    """Height of the image in pixels. Defaults to 256."""
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Position of the camera in the world frame. Defaults to (0.0, 0.0, 1.0)."""
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look at point of the camera in the world frame. Defaults to (0.0, 0.0, 0.0)."""
    mount_to: str | tuple[str, str] | None = None
    """Mount the camera to a specific object or robot. Defaults to None."""
    mount_link: str | tuple[str, str] | None = None
    """Specify the link to mount the camera to. Defaults to None."""
    mount_pos: tuple[float, float, float] | None = None
    """Position of the camera on the mount. Defaults to None."""
    mount_quat: tuple[float, float, float, float] | None = None
    """Quaternion ``(w, x, y, z)`` of the camera on the mount, in the backend's native camera frame: an MJCF
    camera (MuJoCo, SuperDex) looks down its local -Z with +Y up, Isaac Sim's mounted camera looks along +X
    with +Z up, so the same value points a wrist camera differently per backend (``CameraState.quat_world``
    reports where it ended up). Defaults to None."""
    intrinsic: list[list[float]] | None = None
    """Explicit 3x3 pinhole intrinsic [[fx,0,cx],[0,fy,cy],[0,0,1]]. When set, backends that
    support it (sapien2) apply the exact focal lengths / principal point instead of deriving them
    from fov/aperture. Defaults to None (derive from fov)."""

    def __post_init__(self) -> None:
        """A camera needs a positive pixel size and a finite position.

        A zero-sized buffer or a NaN pose fails deep inside the renderer with no reference to the
        camera that caused it.
        """
        from ._validate import finite_triple, positive_finite_or_none, positive_int

        owner = f"{type(self).__name__}({self.name!r})"
        positive_int(owner, "width", self.width)
        positive_int(owner, "height", self.height)
        finite_triple(owner, "pos", self.pos)
        # the intrinsics below divide by these; a zero used to surface as a bare ZeroDivisionError
        for field_name in ("focal_length", "horizontal_aperture"):
            if hasattr(self, field_name):
                positive_finite_or_none(owner, field_name, getattr(self, field_name))
        if getattr(self, "look_at", None) is not None:
            finite_triple(owner, "look_at", self.look_at)


@configclass
class PinholeCameraCfg(BaseCameraCfg):
    """Pinhole camera configuration."""

    focal_length: float = 24.0
    """Perspective focal length (in cm). Defaults to 24.0 cm."""
    focus_distance: float = 400.0
    """Distance from the camera to the focus plane (in m). Defaults to 400.0."""
    horizontal_aperture: float = 20.955
    """Horizontal aperture (in cm). Defaults to 20.955 cm."""
    clipping_range: tuple[float, float] = (0.05, 1e5)
    """Near and far clipping distances (in m). Defaults to (0.05, 1e5)."""
    opencv_distortion: tuple[float, ...] | None = None
    """OpenCV pinhole coefficients in OpenCV order.

    Accepted lengths are 4, 5, 8, 12, or 14. A backend may render an ideal
    pinhole and apply these coefficients in the shared observation pipeline.
    """
    distortion_source: str | None = None
    """Provenance label for ``opencv_distortion`` when present."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.intrinsic is not None:
            matrix = np.asarray(self.intrinsic, dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("intrinsic must be a finite 3x3 matrix")
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0.0, 0.0, 1.0]):
                raise ValueError("intrinsic must have positive focal lengths and last row [0, 0, 1]")
        if self.opencv_distortion is not None:
            coefficients = np.asarray(self.opencv_distortion, dtype=np.float64)
            if coefficients.ndim != 1 or len(coefficients) not in {4, 5, 8, 12, 14}:
                raise ValueError("opencv_distortion must contain 4, 5, 8, 12, or 14 coefficients")
            if not np.isfinite(coefficients).all():
                raise ValueError("opencv_distortion coefficients must be finite")
            self.opencv_distortion = tuple(float(value) for value in coefficients)
        if self.distortion_source is not None and not self.distortion_source.strip():
            raise ValueError("distortion_source must be nonempty when supplied")

    @property
    def vertical_aperture(self) -> float:
        """Vertical aperture (in cm)."""
        return self.horizontal_aperture * self.height / self.width

    @property
    def horizontal_fov(self) -> float:
        """Horizontal field of view, in degrees."""
        return 2 * math.atan(self.horizontal_aperture / (2 * self.focal_length)) / math.pi * 180

    @property
    def vertical_fov(self) -> float:
        """Vertical field of view, in degrees."""
        return 2 * math.atan(self.vertical_aperture / (2 * self.focal_length)) / math.pi * 180

    @property
    def intrinsics(self) -> list[list[float]]:
        """Intrinsics matrix of the camera. Type is 3x3 nested list of floats."""
        if self.intrinsic is not None:
            return self.intrinsic
        fx = self.width * self.focal_length / self.horizontal_aperture
        fy = self.height * self.focal_length / self.vertical_aperture
        cx = self.width * 0.5
        cy = self.height * 0.5
        return [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]


@configclass
class FisheyeCameraCfg(BaseCameraCfg):
    """Calibrated central fisheye camera, in source-image pixel coordinates.

    ``opencv_fisheye`` uses the four OpenCV angular distortion coefficients
    and separate pixel focal lengths. ``ftheta`` uses five coefficients for
    ``radius = k0 + k1*theta + ... + k4*theta**4`` in calibration pixels.
    ``max_fov_rad`` is the total valid field of view, not a half-angle.
    Unsupported backends must fail explicitly rather than render a pinhole.
    """

    projection_model: Literal["opencv_fisheye", "ftheta"] = "opencv_fisheye"
    clipping_range: tuple[float, float] = (0.05, 1e5)
    calibration_size: tuple[int, int] = (0, 0)
    """Width and height of the calibration image in pixels."""
    optical_center: tuple[float, float] = (0.0, 0.0)
    """Principal point ``(cx, cy)`` in calibration-image pixels."""
    focal_length_px: tuple[float, float] | None = None
    """OpenCV ``(fx, fy)`` in calibration pixels; unused for F-theta."""
    coefficients: tuple[float, ...] = ()
    """Four OpenCV angular coefficients or five F-theta radius coefficients."""
    max_fov_rad: float | None = None
    """Total calibrated angular domain in radians, at most 2π."""
    calibration_source: str = "unspecified"
    """Provenance label, such as a source asset or a physical calibration."""
    calibration_source_sha256: str | None = None
    """SHA-256 of the calibration source when it is available."""
    max_fov_source: Literal["specified", "inferred_monotonic"] = "specified"
    """Whether the angular limit was supplied or conservatively inferred."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.projection_model not in {"opencv_fisheye", "ftheta"}:
            raise ValueError("Unsupported fisheye projection model")
        if (
            len(self.calibration_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.calibration_size)
        ):
            raise ValueError("calibration_size must contain two positive integer pixel sizes")
        if len(self.optical_center) != 2 or not all(math.isfinite(float(value)) for value in self.optical_center):
            raise ValueError("optical_center must contain two finite pixel coordinates")
        if self.projection_model == "opencv_fisheye":
            if self.focal_length_px is None or len(self.focal_length_px) != 2 or not all(
                math.isfinite(float(value)) and float(value) > 0 for value in self.focal_length_px
            ):
                raise ValueError("OpenCV fisheye requires two positive finite focal lengths")
            expected = 4
        else:
            if self.focal_length_px is not None:
                raise ValueError("F-theta does not use focal_length_px")
            expected = 5
        if len(self.coefficients) != expected or not all(math.isfinite(float(value)) for value in self.coefficients):
            raise ValueError(f"{self.projection_model} requires {expected} finite coefficients")
        if self.max_fov_rad is None or not math.isfinite(float(self.max_fov_rad)) or not 0 < self.max_fov_rad <= 2 * math.pi:
            raise ValueError("max_fov_rad must be finite and in (0, 2π]")
        if not isinstance(self.calibration_source, str) or not self.calibration_source.strip():
            raise ValueError("calibration_source must be a nonempty string")
        if self.calibration_source_sha256 is not None and (
            len(self.calibration_source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.calibration_source_sha256)
        ):
            raise ValueError("calibration_source_sha256 must be a lowercase SHA-256 digest")
        if self.max_fov_source not in {"specified", "inferred_monotonic"}:
            raise ValueError("max_fov_source must be specified or inferred_monotonic")


REALSENSE_CAMERA = PinholeCameraCfg(
    name="realsense_camera",
    data_types=["rgb", "depth"],
    width=640,
    height=360,
    focal_length=1.88,
    horizontal_aperture=float(2 * 1.88 * math.tan(71.28 / 180 * math.pi / 2)),
)
