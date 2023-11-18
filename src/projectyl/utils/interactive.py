from interactive_pipe import interactive, interactive_pipeline
import numpy as np
import logging
from pathlib import Path
from projectyl.utils.io import Dump
import cv2 as cv

@interactive(frame=(0., [0., 100.]))
def frame_selector(sequence: np.ndarray, frame: float=0., global_params={}) -> np.ndarray:
    frame_idx = int((len(sequence)-1)*frame/100.)
    global_params["frame_idx"] = frame_idx
    return sequence[frame_idx]

@interactive()
def frame_extractor(sequence: np.ndarray, global_params={}) -> np.ndarray:
    frame_idx = global_params.get("frame_idx", 0)
    logging.info(f"{frame_idx} / {len(sequence)} frames")
    return sequence[min(frame_idx, len(sequence)-1)]

@interactive(start=(0., [0., 1.]), end=(1., [0., 1.]))
def trim_seq(sequence, trim_path, start=0, end=1, global_params={}):
    start_idx_ = int(np.round((len(sequence)-1)*start))
    end_idx_ = int(np.round((len(sequence)-1)*end))
    start_idx = min(start_idx_, end_idx_)
    end_idx = max(start_idx_, end_idx_)
    global_params["start"] = start_idx
    global_params["end"] = end_idx
    Dump.save_yaml({"start": start_idx, "end": end_idx}, trim_path)
    return sequence[start_idx, ...], sequence[end_idx, ...]

def trimming(sequence, trim_path):
    start_frame, end_frame = trim_seq(sequence, trim_path)
    return start_frame, end_frame

def interactive_trimming(seq, trim_path: Path):
    interactive_trim_seq = interactive_pipeline(gui="qt", cache=False)(trimming)
    interactive_trim_seq(seq, trim_path)


# Helper to process much smaller raw imagge
@interactive(
    center_x=(0.5, [0., 1.], "cx", ["left", "right"]),
    center_y=(0.5, [0., 1.], "cy", ["up", "down"]),
    size=(10., [6., 13., 0.3], "crop size", ["+", "-"])
)
def crop(image, center_x=0.5, center_y=0.5, size=8.):
    #size is defined in power of 2
    original_shape = image.shape
    if len(image.shape) == 2:
        offset = 0
    elif len(image.shape) == 3:
        channel_guesser_max_size = 4 
        if image.shape[0] <=channel_guesser_max_size: #channel first C,H,W
            offset = 0
        elif image.shape[-1] <=channel_guesser_max_size: #channel last or numpy H,W,C
            offset = 1
    else:
        raise NameError(f"Not supported shape {image.shape}")
    crop_size_pixels =  int(2.**(size)/2.)
    h, w = image.shape[-2-offset], image.shape[-1-offset]
    ar = w/h
    half_crop_h, half_crop_w = crop_size_pixels, int(ar*crop_size_pixels)
    def round(val):
        return int(np.round(val))
    center_x_int = round(half_crop_w +center_x*(w-2*half_crop_w))
    center_y_int = round(half_crop_h +center_y*(h-2*half_crop_h))
    start_x = max(0, center_x_int-half_crop_w)
    start_y = max(0, center_y_int-half_crop_h)
    end_x = min(start_x+2*half_crop_w, w-1)
    end_y = min(start_y+2*half_crop_h, h-1)
    start_x = max(0, end_x-2*half_crop_w)
    start_y = max(0, end_y-2*half_crop_h)
    if offset ==0:
        crop = image[..., start_y:end_y, start_x:end_x]
    if offset==1:
        crop = image[..., start_y:end_y, start_x:end_x, :]
    return cv.resize(crop, (w, h))

def visualize(sequence):
    frame = frame_selector(sequence)
    cropped = crop(frame)
    return cropped

def interactive_visualize(sequence):
    int_viz = interactive_pipeline(gui="qt", cache=False)(visualize)
    int_viz(sequence)