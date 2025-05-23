import numpy as np

from skimage import measure, morphology
from skimage.morphology import disk, opening, closing
from tqdm import tqdm

def Nearest_neighbors(class_mask, min_area=100, morph_kernel_size=3, 
                         fill_holes=True, smooth_boundaries=True, no_data_mask=None):
    if class_mask.max() > 1:
        binary_mask = (class_mask > 127).astype(np.uint8)
    else:
        binary_mask = (class_mask > 0.5).astype(np.uint8)

    if no_data_mask is not None:
        binary_mask = np.where(no_data_mask, binary_mask, 0)

    labeled_mask = measure.label(binary_mask, connectivity=2)
    props = measure.regionprops(labeled_mask)

    filtered_mask = np.zeros_like(binary_mask)
    for prop in props:
        if prop.area >= min_area:
            coords = prop.coords
            filtered_mask[coords[:, 0], coords[:, 1]] = 1

    if fill_holes:
        filtered_mask = morphology.remove_small_holes(
            filtered_mask.astype(bool), 
            area_threshold=min_area // 4
        ).astype(np.uint8)

    kernel = disk(morph_kernel_size)
    if smooth_boundaries:
        filtered_mask = opening(filtered_mask, kernel)

    final_mask = closing(filtered_mask, kernel)

    if no_data_mask is not None:
        final_mask = np.where(no_data_mask, final_mask, 0)

    return final_mask

def postprocess_multiclass_segmentation(label_map, min_area=50, morph_kernel_size=2, 
                                        fill_holes=True, smooth_boundaries=True, no_data_mask=None):
    cleaned_label_map = np.zeros_like(label_map)
    unique_labels = np.unique(label_map)
    print(f"Post-processing {len(unique_labels)} classes...")

    for label in tqdm(unique_labels, desc="Processing classes"):
        if label == 255:
            continue

        class_mask = (label_map == label).astype(np.uint8)

        if no_data_mask is not None:
            class_mask = np.where(no_data_mask, class_mask, 0)

        if label == 0:
            cleaned_label_map[class_mask == 1] = 0
            continue

        cleaned_class_mask = Nearest_neighbors(
            class_mask, 
            min_area=min_area,
            morph_kernel_size=morph_kernel_size,
            fill_holes=fill_holes,
            smooth_boundaries=smooth_boundaries,
            no_data_mask=no_data_mask
        )
        cleaned_label_map[cleaned_class_mask == 1] = label

    if no_data_mask is not None:
        cleaned_label_map[no_data_mask == 0] = 255

    return cleaned_label_map
