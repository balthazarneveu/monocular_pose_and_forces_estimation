from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
from projectyl import root_dir
from typing import Union, List, Optional
import numpy as np
from tqdm import tqdm

MODEL_DIR = root_dir/"model"
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



def segment_frames(input_list: Union[List[Path], np.ndarray], model: Optional[torch.nn.Module]=None):
    if model is None:
        model = load_sam_model()
    mask_generator = SamAutomaticMaskGenerator(model)
    mask_list = []
    for fr_idx in tqdm(range(len(input_list))):
        image = (np.round(input_list[fr_idx])*255).astype(np.uint8)
        masks = mask_generator.generate(image)
        mask_list.append([masks])
    return mask_list