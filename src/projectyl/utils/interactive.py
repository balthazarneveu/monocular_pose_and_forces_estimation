from interactive_pipe import interactive, interactive_pipeline
import numpy as np
import logging
from pathlib import Path
import cv2 as cv
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from typing import List
selected_frames = {}
video_decoder_global = {}


def get_frame(sequence: List[np.ndarray], frame: float, decoder=["cv2", "moviepy"][0]) -> np.ndarray:
    global video_decoder_global
    if isinstance(sequence, list) or isinstance(sequence, tuple) or isinstance(sequence, np.ndarray):
        frame_idx = int((len(sequence)-1)*frame)
        video_decoder_global["frame_idx"] = frame_idx
        print(f"RETURN FRAME [{frame_idx}]")
        return sequence[frame_idx]
    elif isinstance(sequence, str) or isinstance(sequence, Path):
        # ----- LIVE DECODING -----
        # SLOW IN PRACTICE - USE PRELOADING INSTEAD
        seq = video_decoder_global.get("seq", None)
        if decoder == "moviepy":
            # ---------------------------------------------- moviepy
            if seq is None:
                print("LOAD VIDEO SEQUENCE, Live decoding - moviepy")
                seq = VideoFileClip(str(sequence))
                if seq.rotation in (90, 270):  # Support vertical videos
                    seq = seq.resize(seq.size[::-1])
                    seq.rotation = 0
                video_decoder_global["seq"] = seq
            seq = video_decoder_global["seq"]
            frame_time = seq.duration * frame
            frame_idx = int(seq.fps*frame)
            video_decoder_global["frame_idx"] = frame_idx
            grabbed_frame = seq.get_frame(frame_time)
        else:
            # ---------------------------------------------- opencv
            if seq is None:
                print("LOAD VIDEO SEQUENCE, Live decoding")
                seq = cv.VideoCapture(str(sequence))
                video_decoder_global["seq"] = seq
            seq = video_decoder_global["seq"]
            if isinstance(frame, float):
                fps = seq.get(cv.CAP_PROP_FPS)
                frame_idx = int(fps*frame)
            elif isinstance(frame, int):
                frame_idx = int(frame)
            seq.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
            ret, grabbed_frame = seq.read()
            assert ret, f"Cannot read frame {frame_idx}"
            grabbed_frame = cv.cvtColor(grabbed_frame, cv.COLOR_BGR2RGB)
        grabbed_frame = cv.resize(grabbed_frame, None, fx=0.3, fy=0.3)
        grabbed_frame = grabbed_frame/255.
        return grabbed_frame
    else:
        raise NameError(f"Cannot handle type {type(sequence)}")


@interactive(frame=(0., [0., 1.]))
def frame_selector(sequence: np.ndarray, frame: float = 0., global_params={}) -> np.ndarray:
    frame = get_frame(sequence, frame)
    return frame


@interactive()
def frame_extractor(sequence: np.ndarray, global_params={}) -> np.ndarray:
    frame_idx = global_params.get("frame_idx", 0)
    logging.info(f"{frame_idx} / {len(sequence)} frames")
    return sequence[min(frame_idx, len(sequence)-1)]


@interactive(start=(0., [0., 1.]), end=(1., [0., 1.]))
def trim_seq(sequence, start=0, end=1, global_params={}):
    global selected_frames
    start_ratio = min(start, end)
    end_ratio = max(start, end)
    selected_frames = {
        "start_ratio": start_ratio,
        "end_ratio": end_ratio
    }
    if isinstance(sequence, list) or isinstance(sequence, tuple) or isinstance(sequence, np.ndarray):
        logging.debug("TRIM FROM LIST!")
        start_idx_ = int(np.round((len(sequence)-1)*start))
        end_idx_ = int(np.round((len(sequence)-1)*end))
        start_idx = min(start_idx_, end_idx_)
        end_idx = max(start_idx_, end_idx_)
        grabbed_frame_start = sequence[int(start_idx)]
        grabbed_frame_end = sequence[int(end_idx)]  # works with list and np.array
        global_params["start"] = start_idx
        global_params["end"] = end_idx
    else:
        logging.debug("SEQUENCE MODE!")
        grabbed_frame_start = get_frame(sequence, start_ratio)
        grabbed_frame_end = get_frame(sequence, end_ratio)
    return grabbed_frame_start, grabbed_frame_end


def interactive_trimming(seq):
    interactive_trim_seq = interactive_pipeline(gui="auto", cache=False)(trim_seq)
    interactive_trim_seq(seq)


# Helper to process much smaller raw imagge
@interactive(
    center_x=(0.5, [0., 1.], "cx", ["left", "right"]),
    center_y=(0.5, [0., 1.], "cy", ["up", "down"]),
    size=(10., [6., 13., 0.3], "crop size", ["+", "-"])
)
def crop(image, center_x=0.5, center_y=0.5, size=8.):
    # size is defined in power of 2
    if len(image.shape) == 2:
        offset = 0
    elif len(image.shape) == 3:
        channel_guesser_max_size = 4
        if image.shape[0] <= channel_guesser_max_size:  # channel first C,H,W
            offset = 0
        elif image.shape[-1] <= channel_guesser_max_size:  # channel last or numpy H,W,C
            offset = 1
    else:
        raise NameError(f"Not supported shape {image.shape}")
    crop_size_pixels = int(2.**(size)/2.)
    h, w = image.shape[-2-offset], image.shape[-1-offset]
    ar = w/h
    half_crop_h, half_crop_w = crop_size_pixels, int(ar*crop_size_pixels)

    def round(val):
        return int(np.round(val))
    center_x_int = round(half_crop_w + center_x*(w-2*half_crop_w))
    center_y_int = round(half_crop_h + center_y*(h-2*half_crop_h))
    start_x = max(0, center_x_int-half_crop_w)
    start_y = max(0, center_y_int-half_crop_h)
    end_x = min(start_x+2*half_crop_w, w-1)
    end_y = min(start_y+2*half_crop_h, h-1)
    start_x = max(0, end_x-2*half_crop_w)
    start_y = max(0, end_y-2*half_crop_h)
    if offset == 0:
        crop = image[..., start_y:end_y, start_x:end_x]
    if offset == 1:
        crop = image[..., start_y:end_y, start_x:end_x, :]
    return cv.resize(crop, (w, h))


def visualize(sequence):
    frame = frame_selector(sequence)
    cropped = crop(frame)
    return cropped


def interactive_visualize(sequence):
    int_viz = interactive_pipeline(gui="auto", cache=False)(visualize)
    int_viz(sequence)


def trimming_pipeline(sequence):
    start_frame, end_frame = trim_seq(sequence)
    # return start_frame, end_frame
    cropped_start = crop(start_frame)
    cropped_end = crop(end_frame)
    return cropped_start, cropped_end


def live_view(video_path: Path, skip_frames=20, resize=0.3, preload_ram=False, trimming=True):
    """Decode without saving to disk first
    Trim or live view
    """
    # Preload video in RAM
    assert video_path.exists(), f"Video path {video_path} does not exist"
    if preload_ram:
        video = VideoFileClip(str(video_path))
        if video.rotation in (90, 270):  # Support vertical videos
            video = video.resize(video.size[::-1])
            video.rotation = 0
        total_frames = int(video.duration*video.fps)-1
        full_decoded_video_in_ram = [
            cv.resize(video.get_frame(idx*1.0/video.fps), None, fx=resize, fy=resize)/255.
            for idx in tqdm(range(0, total_frames, skip_frames), desc=f"Pre decoding video {video_path.name}")
        ]
        video.close()
    else:
        full_decoded_video_in_ram = str(video_path)
    global video_decoder_global
    video_decoder_global = {}  # reset dict containing current video decoder
    # Live trimming
    if trimming:
        int_viz = interactive_pipeline(gui="auto", cache=False, safe_input_buffer_deepcopy=False)(trimming_pipeline)
        int_viz(full_decoded_video_in_ram, global_params={})
        global selected_frames
        print(f"TRIMMING {video_path.name}:\n{selected_frames}")
        print(len(full_decoded_video_in_ram))
        return selected_frames
    else:
        int_viz = interactive_pipeline(gui="auto", cache=False, safe_input_buffer_deepcopy=False)(visualize)
        int_viz(full_decoded_video_in_ram, global_params={})
