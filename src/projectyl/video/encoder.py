from pathlib import Path

from typing import Optional
from moviepy.editor import ImageSequenceClip
import logging


def encode_debug_figures(input_dir: Path, output_path: Optional[Path] = None, fps=10, resize=0.2) -> Path:
    if isinstance(input_dir, list):
        still_frames = input_dir
    else:
        still_frames = sorted(list(input_dir.glob("*.png")))
    logging.info(f"Encoding {len(still_frames)} frames")
    # Define default output path if not provided
    if output_path is None:
        output_path = input_dir / "output.gif"
    assert not output_path.exists(), f"video file already found {output_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a clip from the images
    clip = ImageSequenceClip([str(img) for img in still_frames], fps=fps)
    clip = clip.resize(resize)
    if output_path.suffix.lower() == ".gif":
        # Write the clip to a GIF file
        clip.write_gif(str(output_path), fps=fps)
    else:
        # Write the clip to a video file MP4/MOV/AVI
        clip.write_videofile(str(output_path), fps=fps)
    return output_path
