from interactive_pipe import interactive, interactive_pipeline
from projectyl.utils.io import Image
import numpy as np
import logging
from pathlib import Path
import cv2 as cv
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from typing import List, Union, Tuple, Optional
selected_frames = {}


def get_frame(sequence: List[np.ndarray], frame: float, decoder=["cv2", "moviepy"][0], video_decoder_global={}) -> np.ndarray:
    # global video_decoder_global
    if isinstance(sequence, list) or isinstance(sequence, tuple) or isinstance(sequence, np.ndarray):
        frame_idx = int((len(sequence)-1)*frame)
        video_decoder_global["frame_idx"] = frame_idx
        logging.info(f"RETURN FRAME [{frame_idx}]")
        if isinstance(sequence[0], str) or isinstance(sequence[0], Path):
            return Image.load(Path(sequence[frame_idx]))/255.
        else:
            return sequence[frame_idx]
    elif isinstance(sequence, str) or isinstance(sequence, Path):
        # ----- LIVE DECODING -----
        # SLOW IN PRACTICE - USE PRELOADING INSTEAD
        seq_key = str(sequence)
        seq = video_decoder_global.get(seq_key, None)
        if decoder == "moviepy":
            # ---------------------------------------------- moviepy
            if seq is None:
                print("LOAD VIDEO SEQUENCE, Live decoding - moviepy")
                seq = VideoFileClip(str(sequence))
                if seq.rotation in (90, 270):  # Support vertical videos
                    seq = seq.resize(seq.size[::-1])
                    seq.rotation = 0
                video_decoder_global[seq_key] = seq
            seq = video_decoder_global[seq_key]
            frame_time = seq.duration * frame
            frame_idx = int(seq.fps*frame)
            video_decoder_global["frame_idx"] = frame_idx
            grabbed_frame = seq.get_frame(frame_time)
        else:
            # ---------------------------------------------- opencv
            if seq is None:
                print("LOAD VIDEO SEQUENCE, Live decoding")
                seq = cv.VideoCapture(str(sequence))
                video_decoder_global[seq_key] = seq
            seq = video_decoder_global[seq_key]
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
    frame = get_frame(sequence, frame, video_decoder_global=global_params)
    return frame


@interactive()
def frame_extractor(sequence: np.ndarray, global_params={}) -> np.ndarray:
    frame_idx = global_params.get("frame_idx", 0)
    logging.info(f"{frame_idx} / {len(sequence)} frames")
    return sequence[min(frame_idx, len(sequence)-1)]


@interactive(start=(0., [0., 1.]), end=(1., [0., 1.]))
def trim_seq(sequence, unique_seq_name, start=0, end=1, global_params={}):
    global selected_frames
    start_ratio = min(start, end)
    end_ratio = max(start, end)
    selected_frames[unique_seq_name] = {
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


@interactive(
    center_x=(0.5, [0., 1.], "cx", ["left", "right"]),
    center_y=(0.5, [0., 1.], "cy", ["up", "down"]),
    size=(11., [6., 13., 0.3], "crop size", ["+", "-"])
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
    MAX_ALLOWED_SIZE = 512
    w_resize = int(min(MAX_ALLOWED_SIZE, w))
    h_resize = int(w_resize/w*h)
    h_resize = int(min(MAX_ALLOWED_SIZE, h_resize))
    w_resize = int(h_resize/h*w)
    return cv.resize(crop, (w_resize, h_resize), interpolation=cv.INTER_NEAREST)


def visualize(sequence):
    frame = frame_selector(sequence)
    cropped = crop(frame)
    return cropped


def interactive_visualize(sequence: Union[Path, List[np.ndarray]]):
    int_viz = interactive_pipeline(gui="auto", cache=True)(visualize)
    int_viz(sequence)


def trimming_pipeline(sequence: Union[Path, List[np.ndarray]], unique_seq_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """PIPELINE TRIM

    Args:
        sequence (Union[Path, List[np.ndarray]]): input sequence (path or list of loaded arrays)
        unique_seq_name (str): Required for multiprocessing

    Returns:
        Tuple[np.ndarray, np.ndarray]: start frame, end frame of the trimmed sequence
    """
    start_frame, end_frame = trim_seq(sequence, unique_seq_name)
    cropped_start = crop(start_frame)
    cropped_end = crop(end_frame)
    return cropped_start, cropped_end


def interactive_trimming(sequence: Union[Path, List[np.ndarray]], unique_seq_name: Optional[str] = None) -> dict:
    """Trim a sequence interactively
    Returns a dict with start_ratio, end_ratio
    Compatible with multiprocessing

    Args:
        sequence (Union[Path, List[np.ndarray]]): path or list of pre-loaded arrays
        unique_seq_name (str): unique identifier for multiprocessing (use video name)

    Returns:
        dict: dict containing trimming info start_ratio, end_ratio
    """
    if unique_seq_name is None and isinstance(sequence, Path):
        unique_seq_name = sequence.name
    assert unique_seq_name is not None, f"{unique_seq_name} is None"
    interactive_trim_seq = interactive_pipeline(gui="auto", cache=True)(trimming_pipeline)
    interactive_trim_seq(sequence, unique_seq_name)
    global selected_frames
    trim_info = selected_frames[unique_seq_name]
    selected_frames.pop(unique_seq_name)  # Remove info
    return trim_info


def live_view(video_path: Path,
              skip_frames: int = 20, resize: float = 0.3, preload_ram: Optional[bool] = False,  # RAM PRELOADING
              trimming: Optional[bool] = True) -> dict:
    """Decode without saving to disk first
    Trim or live view

    Args:
        video_path (Path): video path to the video
        skip_frames (int, optional): skip frames for pre RAM decoding. Defaults to 20.
        resize (float, optional): Resize frames during pre- RAM decoding. Defaults to 0.3.
        preload_ram (Optional[bool], optional): Preload to RAM, slower at first, faster at live view.
        Defaults to False.
        trimming (Optional[bool], optional): Trim (start, end) if True
        Single frame view if False.
        Defaults to True.

    Returns:
        dict: information on trimming
    """
    # Preload video in RAM
    if isinstance(video_path, str) or isinstance(video_path, Path):
        video_path = Path(video_path)
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
    elif isinstance(video_path, list) or isinstance(video_path, tuple):
        if preload_ram:
            print("PRELOADING RAM")
            full_decoded_video_in_ram = np.array(
                [Image.load(Path(img))/255. for img in tqdm(
                    video_path, desc=f"Loading images {Path(video_path[0]).parent.parent.name}")])
        else:
            full_decoded_video_in_ram = video_path

    selected_frames = {}
    if trimming:
        selected_frames = interactive_trimming(full_decoded_video_in_ram, video_path.name)
    else:
        interactive_visualize(full_decoded_video_in_ram)
    return selected_frames
