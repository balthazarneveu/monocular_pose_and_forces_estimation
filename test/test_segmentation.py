from segment_anything import SamAutomaticMaskGenerator
from projectyl.algo.segmentation import load_sam_model
from projectyl import root_dir


# Basic check on sample image
def test_sam(image_path=root_dir/"samples"/"sample_ball.jpg"):
    from projectyl.utils.io import Image
    image = Image.load(image_path)
    sam = load_sam_model()
    mask_generator = SamAutomaticMaskGenerator(sam)
    _masks = mask_generator.generate(image)
    # import matplotlib.pyplot as plt
    # from projectyl.utils.segmentation_masks import show_annotation
    # plt.figure()
    # plt.imshow(image)
    # show_annotation(masks)
    # plt.axis('off')
    # plt.show()