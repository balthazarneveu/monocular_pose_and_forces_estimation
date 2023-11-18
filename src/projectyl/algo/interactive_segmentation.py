from projectyl.algo.segmentation import load_sam_model
from segment_anything import SamAutomaticMaskGenerator
from interactive_pipe import interactive, interactive_pipeline
import numpy as np
import cv2 as cv
from projectyl.utils.interactive import frame_selector, frame_extractor


@interactive(mask_index=(0, [0, 50]))
def display_masks(image, seg_list, mask_index=0):
    seg_list_sorted = sorted(seg_list, key=(lambda x: x['area']), reverse=True)
    seg = seg_list_sorted[min(mask_index, len(seg_list)-1)]
    mask_selected = np.array(seg["segmentation"]).astype(np.uint8)
    masked_img = cv.bitwise_and(image, image, mask=mask_selected)
    return masked_img

def sam_pipeline(sequence, masks):
    frame = frame_selector(sequence)
    mask = frame_extractor(masks)
    overlay = display_masks(frame, mask)
    return frame, overlay

def interactive_sam(sequence, masks):
    interactive_sam_pipeline = interactive_pipeline(gui="qt", cache=False)(sam_pipeline)
    interactive_sam_pipeline(sequence, masks)
