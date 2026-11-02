"""Shared image-space rendering support for calibrated OpenCV pinhole lenses."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=64)
def _opencv_pinhole_map(
    width: int,
    height: int,
    intrinsic_values: tuple[float, ...],
    distortion_values: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build inverse image maps: distorted output pixels sample ideal pinhole pixels."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV pinhole camera distortion requires opencv-python") from exc

    intrinsic = np.asarray(intrinsic_values, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion_values, dtype=np.float64)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    ideal_pixels = cv2.undistortPoints(pixels, intrinsic, distortion, P=intrinsic)
    maps = ideal_pixels.reshape(height, width, 2).astype(np.float32)
    map_x, map_y = maps[..., 0], maps[..., 1]
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return map_x, map_y, valid


def opencv_pinhole_valid_mask(camera, *, device=None) -> torch.Tensor:
    """Return pixels whose inverse pinhole-distortion map stays in the image."""
    if camera.opencv_distortion is None or not any(value != 0 for value in camera.opencv_distortion):
        return torch.ones((camera.height, camera.width), dtype=torch.bool, device=device)
    intrinsic = tuple(float(value) for row in camera.intrinsics for value in row)
    _, _, valid = _opencv_pinhole_map(
        camera.width,
        camera.height,
        intrinsic,
        tuple(float(value) for value in camera.opencv_distortion),
    )
    return torch.as_tensor(valid, dtype=torch.bool, device=device)


def apply_opencv_pinhole_distortion(rgb: torch.Tensor, camera) -> torch.Tensor:
    """Apply an OpenCV pinhole lens model to a ``(N,H,W,3)`` RGB tensor.

    The renderer generates an ideal pinhole image. Each distorted output pixel
    is then sampled from the corresponding ideal pinhole coordinate computed
    with OpenCV's inverse point model. Applying this in the shared task
    observation hook keeps lens behavior identical across physics backends.
    """
    if camera.opencv_distortion is None or not any(value != 0 for value in camera.opencv_distortion):
        return rgb
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB must have shape (N,H,W,3), got {tuple(rgb.shape)}")
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    if (width, height) != (camera.width, camera.height):
        raise ValueError(
            f"Camera {camera.name!r} image shape {(width, height)} does not match config "
            f"{(camera.width, camera.height)}"
        )
    intrinsic = tuple(float(value) for row in camera.intrinsics for value in row)
    map_x, map_y, _ = _opencv_pinhole_map(
        width,
        height,
        intrinsic,
        tuple(float(value) for value in camera.opencv_distortion),
    )
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV pinhole camera distortion requires opencv-python") from exc
    output = [
        cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for frame in rgb.detach().cpu().numpy()
    ]
    return torch.as_tensor(np.stack(output), device=rgb.device, dtype=rgb.dtype)
