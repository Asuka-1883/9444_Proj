"""Read the full observed images for a GC trajectory sample."""
from operator import index

import numpy as np
from PIL import Image


def read_history_images(store, sample):
    """Return the sample's observed frames as RGB uint8 [T, H, W, 3]."""
    images = []
    for frame_id in sample['obs_frame_ids']:
        path = store.frame_paths[index(frame_id)]
        with Image.open(path) as image:
            images.append(np.array(image.convert('RGB'), dtype=np.uint8))
    return np.stack(images, axis=0)
