import numpy as np
from interactive_pipe import interactive, interactive_pipeline
from scipy.ndimage import gaussian_filter
import cv2 as cv
@interactive(frame=(0., [0., 100.]))
def frame_selector(sequence: np.ndarray, frame: float=0.) -> np.ndarray:
    frame_idx = int((len(sequence)-1)*frame/100.)
    return sequence[frame_idx]

def bg_estimator(sequence: np.ndarray) -> np.ndarray:
    avg = np.median(sequence, axis=0)
    return avg

@interactive(sigma=(1., [1., 15.]))
def bg_sub(frame, bg, sigma=1.):
    # @TODO: replace by a fancier detector
    diff = frame-bg
    diff_blurred = gaussian_filter(diff, sigma=sigma)
    return np.abs(diff_blurred)

def detect(diff, frame):
    # Naively detect the argmax
    # diff_y = np.sum(diff, axis=-1)
    diff_y = diff[..., 0] # take the red instead of (R+G+B)/3
    ind = np.unravel_index(np.argmax(diff_y, axis=None), diff_y.shape)
    marked_frame = frame.copy()
    marked_frame = cv.circle(marked_frame, ind[::-1], radius=10, thickness=5, color=(255, 0, 0))
    return marked_frame

@interactive_pipeline(gui="qt", cache=True)
def bg_substract_interactive(sequence):
    bg = bg_estimator(sequence)
    frame = frame_selector(sequence)
    diff = bg_sub(frame, bg)
    marked_frame = detect(diff, frame)
    return frame, bg, diff, marked_frame

def bg_substract(sequence:np.ndarray, interactive=False):
    if interactive:
        bg_substract_interactive(sequence)
    bg = bg_estimator(sequence)
    bg_sub(sequence, bg)