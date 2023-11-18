from projectyl.algo.segmentation import load_sam_model
from segment_anything import SamAutomaticMaskGenerator
from interactive_pipe import interactive, interactive_pipeline
import numpy as np
from projectyl.utils.interactive import frame_selector

def load_mask_gen():
    model = load_sam_model()
    mask_generator = SamAutomaticMaskGenerator(model)
    return mask_generator


def segment_something(mask_generator, frame):
    image = (np.round(frame)*255).astype(np.uint8)
    masks = mask_generator.generate(image)
    print(masks)


def sam_pipeline(sequence):
    mg = load_mask_gen()
    frame = frame_selector(sequence)
    segment_something(mg, frame)
    return frame

def interactive_sam(sequence):
    interactive_pipeline(gui="qt", cache=True)(sam_pipeline)
    sam_pipeline(sequence)
