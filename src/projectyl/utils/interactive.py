from interactive_pipe import interactive
import numpy as np

@interactive(frame=(0., [0., 100.]))
def frame_selector(sequence: np.ndarray, frame: float=0.) -> np.ndarray:
    frame_idx = int((len(sequence)-1)*frame/100.)
    return sequence[frame_idx]
