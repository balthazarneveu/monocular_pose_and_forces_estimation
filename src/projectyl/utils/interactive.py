from interactive_pipe import interactive
import numpy as np
import logging

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