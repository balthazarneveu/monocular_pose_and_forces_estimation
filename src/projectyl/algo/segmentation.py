from pathlib import Path
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import torch
ROOT_DIR = Path(__file__).parent.parent.parent.parent
MODEL_DIR = ROOT_DIR/"model"
MODEL_DICT = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}
def load_sam_model(model_directory: Path =MODEL_DIR, model_name: str="vit_b", device: str="cuda") -> torch.nn.Module:
    """Load SAM (segment anything model) from FAIR
    https://github.com/facebookresearch/segment-anything

    Args:
        model_directory (Path, optional): directory containing all models. Defaults to MODEL_DIR.
        model_name (str, optional): model name vit_h is big, vit_b is small. Defaults to "vit_b".
        device (str, optional): torch device. Defaults to "cuda". Requirement: run on GPU

    Returns:
        torch.nn.Module: SAM instance
    """
    assert model_directory.exists()
    model_path = model_directory/MODEL_DICT.get(model_name, MODEL_DICT["vit_b"])
    assert model_path.exists(), "download models from https://github.com/facebookresearch/segment-anything"
    sam = sam_model_registry[model_name](checkpoint=model_path)
    sam.to(device)
    return sam


# Basic check on sample image
def check_sam(image_path=ROOT_DIR/"samples"/"sample_ball.jpg"):
    from projectyl.utils.segmentation_masks import show_annotation
    from projectyl.utils.io import Image
    import matplotlib.pyplot as plt
    image = Image.load(image_path)
    sam = load_sam_model()
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    plt.figure()
    plt.imshow(image)
    show_annotation(masks)
    plt.axis('off')
    plt.show() 



if __name__ == "__main__":
    check_sam()
    