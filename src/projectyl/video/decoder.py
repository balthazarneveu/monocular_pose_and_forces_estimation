from pathlib import Path
from moviepy.editor import VideoFileClip
from typing import Optional

def save_video_frames(input_path: Path, output_folder: Path, trim=None, resize: Optional[float]=None):
    video = VideoFileClip(str(input_path))
    if trim is not None:
        video = video.subclip(*trim)
    if resize is not None:
        video = video.resize(resize)
    if video.rotation in (90, 270): # Support vertical videos
        # https://github.com/Zulko/moviepy/issues/586
        video = video.resize(video.size[::-1])
        video.rotation = 0
    video_name = input_path.stem
    video.write_images_sequence(str(output_folder/f'{video_name}_%05d.jpg'), logger='bar')