"""Small PyAV helpers shared by ICL scene-video probes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class RGBVideoWriter:
    """Write fixed-rate RGB frames to H.264 MP4, padding odd edges for yuv420p."""

    def __init__(self, path: Path, width: int, height: int, fps: int = 30):
        if min(width, height, fps) < 1:
            raise ValueError("Video width, height and fps must be positive")
        try:
            import av
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("MP4 writing requires PyAV and NumPy; install the RoboVerse `icl` extra") from exc

        self._av = av
        self._np = np
        self.width, self.height = width, height
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width + width % 2
        self.stream.height = height + height % 2
        self.stream.pix_fmt = "yuv420p"
        self._closed = False

    def append(self, rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed video writer")
        if rgb.dtype != self._np.uint8 or rgb.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected RGB uint8 frames of {(self.height, self.width, 3)}, got {rgb.shape} {rgb.dtype}"
            )
        if self.width % 2 or self.height % 2:
            rgb = self._np.pad(rgb, ((0, self.height % 2), (0, self.width % 2), (0, 0)), mode="edge")
        frame = self._av.VideoFrame.from_ndarray(self._np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self._closed:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self._closed = True


def write_preview_gif(
    source: Path,
    target: Path,
    *,
    max_size: int = 384,
    target_fps: int = 15,
    max_frames: int = 60,
) -> tuple[int, int, int, float]:
    """Create a small looping GIF from an existing short MP4 scene preview.

    At most four seconds are retained at the requested frame rate. Aspect ratio
    and the original MP4 are preserved; this is only a Markdown preview copy.
    """
    import math

    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "GIF preview generation requires PyAV and Pillow; install the RoboVerse `icl` extra"
        ) from exc

    if min(max_size, target_fps, max_frames) < 1:
        raise ValueError("GIF size, frame rate and frame cap must be positive")
    frames = []
    with av.open(str(source), mode="r") as container:
        streams = [stream for stream in container.streams if stream.type == "video"]
        if not streams:
            raise ValueError(f"Preview file has no video stream: {source}")
        stream = streams[0]
        stream_rate = stream.average_rate or stream.base_rate
        source_fps = float(stream_rate) if stream_rate is not None else float(target_fps)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"Preview file has an invalid frame rate: {source}")
        sample_step = max(1, round(source_fps / target_fps))
        output_fps = source_fps / sample_step
        for index, frame in enumerate(container.decode(stream)):
            if index % sample_step:
                continue
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", frames[0].size if frames else image.size)
            offset = ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2)
            canvas.paste(image, offset)
            frames.append(canvas)
            if len(frames) >= max_frames:
                break
    if len(frames) < 2:
        raise ValueError(f"Preview file has fewer than two usable frames: {source}")

    palette_sheet = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for index, frame in enumerate(frames):
        palette_sheet.paste(frame, (0, index * frames[0].height))
    palette = palette_sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed_frames = [frame.quantize(palette=palette) for frame in frames]
    target.parent.mkdir(parents=True, exist_ok=True)
    indexed_frames[0].save(
        target,
        save_all=True,
        append_images=indexed_frames[1:],
        duration=max(20, round(1000 / output_fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return len(frames), frames[0].width, frames[0].height, output_fps
