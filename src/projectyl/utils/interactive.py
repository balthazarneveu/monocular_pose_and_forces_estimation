from interactive_pipe import interactive, interactive_pipeline
import numpy as np
import logging
from pathlib import Path
from projectyl.utils.io import Dump

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
