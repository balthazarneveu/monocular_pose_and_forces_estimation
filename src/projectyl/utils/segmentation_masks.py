import numpy as np
import matplotlib.pyplot as plt
import logging

def annotation_overlay(anns: dict, image: np.ndarray=None, ax: plt.Axes =None, show: bool=True):
    """Add annotation overlay on top of current figure (or image)
    """
    if ax is None:
        plt.figure()
    if image is not None:
        plt.imshow(image)
    if len(anns) == 0:
        logging.info("no maks found")
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    ax.set_axis_off()
    if show:
        plt.show()