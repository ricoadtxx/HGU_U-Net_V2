import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import cv2

from config import TIMESTAMP, EPOCHS, CLASS_NAMES, IMAGE_PATCH_SIZE
from resampling import ResampleTransform
from tf_color import ColorTransferTransform

def plot_training_results(history, epochs=EPOCHS):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['jaccard_coef'], label='Training MIoU', color='green')
    plt.plot(history.history['val_jaccard_coef'], label='Validation MIoU', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('MIoU')
    plt.title('Training & Validation MIoU')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='purple')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid()

    filename = f"report/training_results_{epochs}e_{TIMESTAMP}.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafik hasil pelatihan disimpan sebagai {filename}")
    plt.show()

def visualize_sample_data(X, y, validity_mask=None, num_samples=5, class_names=CLASS_NAMES):
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)

    fig_cols = 3 if validity_mask is not None else 2
    plt.figure(figsize=(5 * fig_cols, num_samples * 5))
    
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, fig_cols, i*fig_cols + 1)
        plt.imshow(X[idx])
        plt.title(f"Sample {idx} - Image")
        plt.axis('off')

        plt.subplot(num_samples, fig_cols, i*fig_cols + 2)
        if y[idx].ndim == 3 and y[idx].shape[-1] > 1:
            masks_display = np.argmax(y[idx], axis=-1)
        else:
            masks_display = y[idx]
            
        cmap = plt.get_cmap('tab10', np.max(masks_display) + 1 if class_names else None)
        im = plt.imshow(masks_display, cmap=cmap)
        plt.title(f"Sample {idx} - Masks")
        plt.axis('off')
        
        if class_names and fig_cols == 2:
            cbar = plt.colorbar(im, ticks=np.arange(len(class_names)))
            cbar.ax.set_yticklabels(class_names)

        if validity_mask is not None:
            plt.subplot(num_samples, fig_cols, i*fig_cols + 3)
            plt.imshow(validity_mask[idx], cmap='gray')
            plt.title(f"Sample {idx} - Validity Mask")
            plt.axis('off')

    plt.tight_layout()
    filename = f"report/visualisation_sample_{TIMESTAMP}.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafik data sampel disimpan sebagai {filename}")
    plt.show()

def plot_augmentation(image, mask, validity_mask=None, image_patch_size=IMAGE_PATCH_SIZE):
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask_single_channel = np.argmax(mask, axis=-1).astype(np.uint8)
    else:
        mask_single_channel = mask.astype(np.uint8)
    
    augmentations = [
        ("Flip Horizontal", A.HorizontalFlip(p=1.0)),
        ("Flip Vertical", A.VerticalFlip(p=1.0)),
        ("Rotation", A.Rotate(limit=45, p=1.0)),
        ("Brightness Contrast", A.RandomBrightnessContrast(brightness_limit=-0.3, contrast_limit=0.1, p=1.0)),
        ("Elastic Transform", A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0)),
        ("Gaussian Blur", A.GaussianBlur(blur_limit=(3, 5), p=0.3)),
        ("Random Scale", A.RandomScale(scale_limit=0.3, p=1.0)),
        ("Resampling", ResampleTransform(scale_limit=0.5, interpolation=cv2.INTER_NEAREST, p=1.0)),
        ("Color Transfer", ColorTransferTransform(reference_dir='references', p=0.3)),
    ]

    fig_cols = 3 if validity_mask is not None else 2
    plt.figure(figsize=(5 * fig_cols, len(augmentations) * 5))

    for i, (name, aug) in enumerate(augmentations):
        additional_targets = {'validity_mask': 'mask'} if validity_mask is not None else {}
        
        transform = A.Compose(
            [aug, A.Resize(image_patch_size, image_patch_size)],
            additional_targets=additional_targets
        )
        
        if validity_mask is not None:
            augmented = transform(image=image, mask=mask_single_channel, validity_mask=validity_mask)
            aug_image, aug_mask, aug_valid = augmented['image'], augmented['mask'], augmented['validity_mask']
        else:
            augmented = transform(image=image, mask=mask_single_channel)
            aug_image, aug_mask = augmented['image'], augmented['mask']
            aug_valid = None
        
        plt.subplot(len(augmentations), fig_cols, i*fig_cols + 1)
        plt.imshow(aug_image)
        plt.title(f"{name} - Gambar")
        plt.axis('off')

        plt.subplot(len(augmentations), fig_cols, i*fig_cols + 2)
        plt.imshow(aug_mask, cmap='tab10')
        plt.title(f"{name} - Masking")
        plt.axis('off')

        if validity_mask is not None:
            plt.subplot(len(augmentations), fig_cols, i*fig_cols + 3)
            plt.imshow(aug_valid, cmap='gray')
            plt.title(f"{name} - Validity Mask")
            plt.axis('off')

    plt.tight_layout()
    filename = f"report/augmentasi_sample_{TIMESTAMP}.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafik augmentasi data disimpan sebagai {filename}")
    plt.show()

def show_augmentation(image_dataset, mask_dataset, validity_masks=None, num_samples=1, image_patch_size=IMAGE_PATCH_SIZE):
    indices = np.random.choice(len(image_dataset), min(num_samples, len(image_dataset)), replace=False)
    
    for idx in indices:
        image = image_dataset[idx]
        mask = mask_dataset[idx]
        validity_mask = validity_masks[idx] if validity_masks is not None else None
        plot_augmentation(image, mask, validity_mask, image_patch_size)

def visualize_predict(model, X_test, y_test, num_samples=5):
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)

    plt.figure(figsize=(15, num_samples * 5))

    for i, idx in enumerate(indices):
        # Gambar input
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(X_test[idx])
        plt.title(f"Sample {idx} - Input")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 2)
        if y_test[idx].ndim == 3 and y_test[idx].shape[-1] > 1:
            ground_truth = np.argmax(y_test[idx], axis=-1)
        else:
            ground_truth = y_test[idx]
        plt.imshow(ground_truth, cmap='tab10')
        plt.title(f"Sample {idx} - Ground Truth")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 3)
        input_img = np.expand_dims(X_test[idx], axis=0)
        pred_mask = model.predict(input_img)
        pred_mask = np.argmax(pred_mask[0], axis=-1)
        plt.imshow(pred_mask, cmap='tab10')
        plt.title(f"Sample {idx} - Prediction")
        plt.axis('off')

    plt.tight_layout()
    filename = f"report/prediksi_results_{TIMESTAMP}.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafik hasil prediksi disimpan sebagai {filename}")
    plt.show()