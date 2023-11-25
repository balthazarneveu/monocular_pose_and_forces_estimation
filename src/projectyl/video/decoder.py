from pathlib import Path
from typing import Optional
import logging
import cv2
from projectyl.utils.io import Image
from projectyl.video.props import FRAMES, THUMBS, FRAME_IDX, TS, FOLDER, PATH_LIST, SIZE


def save_video_frames_moviepy(input_path: Path, output_folder: Path, trim=None, resize: Optional[float] = None):
    from moviepy.editor import VideoFileClip
    with VideoFileClip(str(input_path)) as video:
        if resize is not None:
            video = video.resize(resize)
        if video.rotation in (90, 270):  # Support vertical videos
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
            if end is not None and frame_idx > end:
                logging.info(f"LAST FRAME REACHED! {frame_idx}>{end}")
                break
            if start is not None and frame_idx <= start:
                continue
            Image.write(output_folder/f'{video_name}_{frame_idx:05d}.jpg', frame)


def save_video_frames(input_path: Path, output_folder: Path, trim=None, resize: Optional[float] = None):
    video_name = input_path.stem
    video = cv2.VideoCapture(str(input_path))
    total_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    start, end = None, None

    if trim is not None:
        assert len(trim) == 2
        start, end = trim
        if start is not None:
            start = int(start*total_length)
        if end is not None:
            end = int(end*total_length)
    if (video.isOpened() == False):
        logging.warning("Error opening video stream or file")
    frame_idx = -1
    pth_list = []
    frame_indices = []
    frame_ts = []
    while (video.isOpened()):
        # Capture frame-by-frame
        ret, frame = video.read()
        frame_idx += 1
        # if frame_idx % 10 == 0:
        #     print(frame_idx)
        if not ret:
            break
        if end is not None and frame_idx > end:
            logging.info(f"LAST FRAME REACHED! {frame_idx}>{end}")
            break
        if start is not None and frame_idx <= start:
            continue
        logging.info(f"{frame_idx}, {frame.shape}")
        original_size = frame.shape[:2]
        rs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            rs_frame = cv2.resize(rs_frame, None, fx=resize, fy=resize)
            rs_frame_size = frame
        else:
            rs_frame_size = original_size
        pth = output_folder/f'{video_name}_{frame_idx:05d}.jpg'
        frame_indices.append(frame_idx)
        frame_ts.append(frame_idx/fps)
        pth_list.append(str(pth))
        Image.write(pth, rs_frame)
    sample_config_file = {
        "start_frame": start,
        "end_frame": end,
        "total_frames": total_length,
        "fps": fps,
        FRAMES: {
            SIZE: original_size,
            FRAME_IDX: frame_indices,
            TS: frame_ts,
        },
        THUMBS: {
            SIZE: rs_frame_size,
            FRAME_IDX: frame_indices,
            TS: frame_ts,
            FOLDER: str(output_folder),
            PATH_LIST: pth_list
        }
    }
    return sample_config_file
