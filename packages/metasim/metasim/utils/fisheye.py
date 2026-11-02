"""Central fisheye projection and cubemap sampling in OpenGL camera coordinates."""

from __future__ import annotations

import math

import numpy as np


def invertible_fisheye_fov(model: str, coefficients, requested_fov: float) -> float:
    """Return a total FOV strictly inside the first radial-polynomial turning point."""
    coeff = np.asarray(coefficients, dtype=np.float64)
    if not np.isfinite(coeff).all() or not np.isfinite(requested_fov) or not 0 < requested_fov <= 2 * math.pi:
        raise ValueError("Fisheye coefficients and field of view must be finite, with 0 < FOV <= 2*pi")
    if model == "opencv_fisheye" and len(coeff) == 4:
        roots = np.polynomial.polynomial.polyroots([1.0, *(coeff * [3, 5, 7, 9])])
        candidates = [math.sqrt(root.real) for root in roots if abs(root.imag) < 1e-8 and root.real > 0]
    elif model == "ftheta" and len(coeff) == 5:
        if coeff[1] <= 0:
            raise ValueError("F-theta radius must initially increase with incidence angle")
        roots = np.polynomial.polynomial.polyroots(coeff[1:] * [1, 2, 3, 4])
        candidates = [float(root.real) for root in roots if abs(root.imag) < 1e-8 and root.real > 0]
    else:
        raise ValueError(f"Invalid fisheye model or coefficient count: {model!r}")
    if candidates and min(candidates) <= requested_fov / 2:
        return 2 * min(candidates) * (1 - 1e-5)
    return float(requested_fov)


def validate_fisheye_domain(camera) -> None:
    """Reject a requested FOV that crosses an ambiguous radial projection."""
    supported = invertible_fisheye_fov(camera.projection_model, camera.coefficients, camera.max_fov_rad)
    if supported < camera.max_fov_rad:
        raise ValueError(
            f"Camera {camera.name}: max_fov_rad={camera.max_fov_rad:.6g} exceeds the invertible "
            f"fisheye domain; use a value below {supported:.6g}."
        )


def fisheye_rays(camera) -> tuple[np.ndarray, np.ndarray]:
    """Return unit rays (H,W,3) and valid-pixel mask for one calibrated lens.

    Pixels use center sampling, +X right, +Y up, and -Z forward, matching USD,
    MuJoCo, Genesis and Newton's camera frames. Pixels outside the explicit
    invertible angular range have zero rays and a false mask.
    """
    validate_fisheye_domain(camera)
    width, height = camera.width, camera.height
    native_width, native_height = camera.calibration_size
    cx, cy = camera.optical_center
    u = (np.arange(width, dtype=np.float64) + 0.5) * native_width / width
    v = (np.arange(height, dtype=np.float64) + 0.5) * native_height / height
    x, y = np.meshgrid(u - cx, -(v - cy))
    if camera.projection_model == "opencv_fisheye":
        fx, fy = camera.focal_length_px
        x, y = x / fx, y / fy
        k1, k2, k3, k4 = camera.coefficients

        def radial(theta):
            theta2 = theta * theta
            return theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
    else:
        k0, k1, k2, k3, k4 = camera.coefficients

        def radial(theta):
            return k0 + theta * (k1 + theta * (k2 + theta * (k3 + theta * k4)))

    radius = np.hypot(x, y)
    max_theta = min(float(camera.max_fov_rad) / 2, math.pi)
    valid = radius <= float(radial(max_theta)) + 1e-5
    lower, upper = np.zeros_like(radius), np.full_like(radius, max_theta)
    for _ in range(30):
        mid = (lower + upper) / 2
        below = radial(mid) < radius
        lower = np.where(below, mid, lower)
        upper = np.where(below, upper, mid)
    theta = (lower + upper) / 2
    if camera.projection_model == "ftheta":
        theta = np.where(radius <= k0, 0.0, theta)
    theta = np.where(radius <= 1e-7, 0.0, theta)
    scale = np.divide(np.sin(theta), radius, out=np.zeros_like(radius), where=radius > 1e-7)
    rays = np.stack((x * scale, y * scale, -np.cos(theta)), axis=-1)
    rays[~valid] = 0.0
    return rays.astype(np.float32), valid


# Each column is the face camera's +X, +Y and +Z axis in the fisheye frame.
CUBEMAP_ROTATIONS = np.asarray([
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],   # +X
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],   # -X
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],   # +Y
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],   # -Y
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],  # +Z
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],    # -Z
], dtype=np.float64)


def cubemap_lookup(rays: np.ndarray, valid: np.ndarray, face_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map unit rays to six 90-degree square face images with pixel-center coordinates."""
    if face_size < 2 or rays.shape != (*valid.shape, 3):
        raise ValueError("Invalid rays, mask or cubemap face size")
    # Rotations map face coordinates to the shared camera frame; columns 2
    # point backward, so the face optical direction is the negated last column.
    forwards = -CUBEMAP_ROTATIONS[:, :, 2]
    scores = rays @ forwards.T
    faces = np.argmax(scores, axis=-1)
    chosen = CUBEMAP_ROTATIONS[faces]
    face_rays = np.einsum("...ji,...j->...i", chosen, rays)
    z = -face_rays[..., 2]
    u = np.divide(face_rays[..., 0], z, out=np.zeros_like(z), where=valid & (z > 0))
    v = np.divide(face_rays[..., 1], z, out=np.zeros_like(z), where=valid & (z > 0))
    return faces, (u + 1) * face_size / 2 - 0.5, (1 - v) * face_size / 2 - 0.5


def sample_cubemap_rgb(images: list[np.ndarray], lookup, valid: np.ndarray) -> np.ndarray:
    """Bilinearly sample RGB faces; invalid pixels are black and flagged by the mask."""
    if len(images) != 6 or any(image.ndim != 3 or image.shape[-1] != 3 for image in images):
        raise ValueError("Expected six RGB cubemap images")
    size = images[0].shape[0]
    if any(image.shape != (size, size, 3) for image in images):
        raise ValueError("Cubemap images must be equal-sized squares")
    faces, u, v = lookup
    result = np.zeros((*valid.shape, 3), dtype=np.float32)
    for index, image in enumerate(images):
        selected = valid & (faces == index)
        if not np.any(selected):
            continue
        x = np.clip(u[selected], 0, size - 1)
        y = np.clip(v[selected], 0, size - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, size - 1), np.minimum(y0 + 1, size - 1)
        wx, wy = (x - x0)[:, None], (y - y0)[:, None]
        result[selected] = (
            image[y0, x0] * (1 - wx) * (1 - wy) + image[y0, x1] * wx * (1 - wy)
            + image[y1, x0] * (1 - wx) * wy + image[y1, x1] * wx * wy
        )
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)
