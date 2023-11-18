from segment_anything import SamAutomaticMaskGenerator
from projectyl.algo.segmentation import load_sam_model
from projectyl import root_dir
from projectyl.utils.io import Image
import matplotlib.pyplot as plt
from projectyl.utils.segmentation_masks import annotation_overlay
    
# Basic check on sample image
def test_sam(image_path=root_dir/"samples"/"sample_ball.jpg", show=False):
    image = Image.load(image_path)
    sam = load_sam_model()
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    annotation_overlay(masks, image=image, show=show)
