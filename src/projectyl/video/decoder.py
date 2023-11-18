from pathlib import Path
from moviepy.editor import VideoFileClip
from typing import Optional
import logging
from projectyl.utils.io import Image

def save_video_frames(input_path: Path, output_folder: Path, trim=None, resize: Optional[float]=None):
    # video = VideoFileClip(str(input_path))
    with VideoFileClip(str(input_path)) as video:
        if resize is not None:
            video = video.resize(resize)
        if video.rotation in (90, 270): # Support vertical videos
            # https://github.com/Zulko/moviepy/issues/586
            video = video.resize(video.size[::-1])
            video.rotation = 0
        video_name = input_path.stem
        
        # video.write_images_sequence(str(output_folder/f'{video_name}_%05d.jpg'), logger='bar') *
        # BUG DUPLCATED FRAMES!
        
        start, end = None, None
        if trim is not None:
            assert len(trim) == 2
            start, end = trim
            if start is not None:
                start = int(start*video.fps)
            if end is not None:
                end = int(end*video.fps)
        for frame_idx, frame in enumerate(video.iter_frames()):
            if end is not None and frame_idx>end:
                logging.info(f"LAST FRAME REACHED! {frame_idx}>{end}")
                break
            if start is not None and frame_idx<= start:
                continue
            Image.write(output_folder/f'{video_name}_{frame_idx:05d}.jpg', frame)