import os
import cv2
import numpy as np
from patchify import patchify
from tensorflow.keras.utils import to_categorical
import albumentations as A

from config import DATA_DIR, DATA_NAME, BATCH_SIZE, IMAGE_PATCH_SIZE, NUM_CLASSES, VALID_THRESHOLD, THRESHOLD
from resampling import ResampleTransform
from tf_color import ColorTransferTransform

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def is_valid_patch(patch, threshold=VALID_THRESHOLD):
    no_data_mask = np.all(patch < 10, axis=-1)
    valid_ratio = 1.0 - (np.sum(no_data_mask) / no_data_mask.size)
    return valid_ratio >= threshold

def create_nodata_mask(image):
    no_data_mask = ~np.all(image < THRESHOLD, axis=-1)
    return no_data_mask.astype(np.uint8)

def load_dataset(valid_threshold=VALID_THRESHOLD):
    image_dataset, mask_dataset, validity_masks = [], [], []

    for image_type in ['images', 'masking']:
        image_extension = 'jpg' if image_type == 'images' else 'png'

        for tile_id in range(1, 10):
            for image_id in range(1, 10):
                image_path = f'{DATA_DIR}{DATA_NAME}/Tile {tile_id}/{image_type}/image_00{image_id}.{image_extension}'
                if not os.path.exists(image_path):
                    print(f"Warning: File not found: {image_path}")
                    continue
                    
                image = cv2.imread(image_path)

                if image is not None:
                    if image_type == 'masking':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    size_x = (image.shape[1] // IMAGE_PATCH_SIZE) * IMAGE_PATCH_SIZE
                    size_y = (image.shape[0] // IMAGE_PATCH_SIZE) * IMAGE_PATCH_SIZE
                    cropped_image = image[:size_y, :size_x]
                    patched_images = patchify(cropped_image, (IMAGE_PATCH_SIZE, IMAGE_PATCH_SIZE, 3), step=IMAGE_PATCH_SIZE)

                    for i in range(patched_images.shape[0]):
                        for j in range(patched_images.shape[1]):
                            patch = patched_images[i, j, 0, :]

                            if is_valid_patch(patch, valid_threshold):
                                if image_type == 'images':
                                    validity_mask = create_nodata_mask(patch)
                                    validity_masks.append(validity_mask)
                                    patch = normalize_image(patch)
                                    image_dataset.append(patch)
                                else:
                                    mask_dataset.append(patch)
    
    print(f"Loaded {len(image_dataset)} valid patches out of total patches")
    return np.array(image_dataset), np.array(mask_dataset), np.array(validity_masks)

def preprocess_masks(mask_dataset, num_classes=NUM_CLASSES):
    colors = {
        'ground': '#A7A8A7', 'hutan': '#15C23B', 'palmoil': '#2E0F0F',
        'urban': '#ED5C0E', 'vegetation': '#66ED45'
    }
    class_colors = {
        k: np.array([int(v.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)])
        for k, v in colors.items()
    }

    def rgb_to_label(label):
        label_segment = np.zeros(label.shape[:2], dtype=np.uint8)
        for i, color in enumerate(class_colors.values()):
            label_segment[np.all(label == color, axis=-1)] = i
        return label_segment

    labels = np.array([rgb_to_label(mask) for mask in mask_dataset])
    return to_categorical(labels, num_classes=num_classes)

def augment_data(image, mask, validity_mask=None):
    if isinstance(mask, np.ndarray) and mask.ndim == 3 and mask.shape[-1] > 1:
        mask = np.argmax(mask, axis=-1).astype(np.uint8)

    additional_targets = {'validity_mask': 'mask'} if validity_mask is not None else {}
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=-0.3, contrast_limit=-0.1, p=1.0),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        ResampleTransform(scale_limit=0.5, interpolation=cv2.INTER_NEAREST, p=1.0),
        ColorTransferTransform(reference_dir='references', p=0.3),  # Tambahkan ini
        A.Resize(IMAGE_PATCH_SIZE, IMAGE_PATCH_SIZE)
    ], additional_targets=additional_targets)

    if validity_mask is not None:
        augmented = transform(image=image, mask=mask, validity_mask=validity_mask)
        return augmented['image'], augmented['mask'], augmented['validity_mask']
    else:
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

def data_generator(images, masks, batch_size=BATCH_SIZE, validity_masks=None):
    while True:
        idx = np.random.permutation(len(images))
        for i in range(0, len(images), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_images, batch_masks = [], []

            for j in batch_idx:
                if validity_masks is not None and j < len(validity_masks):
                    img, mask, _ = augment_data(images[j], masks[j], validity_masks[j])
                else:
                    img, mask = augment_data(images[j], masks[j])

                batch_images.append(img)

                if mask.ndim == 2:
                    mask = to_categorical(mask, num_classes=NUM_CLASSES)
                batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)