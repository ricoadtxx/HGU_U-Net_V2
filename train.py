import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from datetime import datetime
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from patchify import patchify
import albumentations as A
import argparse
from model import multi_unet_model
from losses import jaccard_coef

# Konfigurasi awal
os.environ["SM_FRAMEWORK"] = "tf.keras"
dataset = 'dataset_new/'
dataset_name = 'DBX'
image_patch_size = 256
num_classes = 5

def normalize_image(image):
    """Normalisasi nilai piksel gambar ke [0,1]."""
    return image.astype(np.float32) / 255.0

def is_valid_patch(patch, valid_threshold=0.1):
    no_data_mask = np.all(patch < 10, axis=-1)
    valid_ratio = 1.0 - (np.sum(no_data_mask) / no_data_mask.size)
    return valid_ratio >= valid_threshold

def create_nodata_mask(image, threshold=2):
    no_data_mask = ~np.all(image < threshold, axis=-1)
    return no_data_mask.astype(np.uint8)

def load_dataset(valid_threshold=0.1):
    """Load image dataset dan masking dataset."""
    image_dataset, mask_dataset, validity_masks = [], [], []

    for image_type in ['images', 'masking']:
        image_extension = 'jpg' if image_type == 'images' else 'png'

        for tile_id in range(1, 10):
            for image_id in range(1, 10):
                image_path = f'{dataset}{dataset_name}/Tile {tile_id}/{image_type}/image_00{image_id}.{image_extension}'
                image = cv2.imread(image_path)

                if image is not None:
                    if image_type == 'masking':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    size_x = (image.shape[1] // image_patch_size) * image_patch_size
                    size_y = (image.shape[0] // image_patch_size) * image_patch_size
                    cropped_image = image[:size_y, :size_x]
                    patched_images = patchify(cropped_image, (image_patch_size, image_patch_size, 3), step=image_patch_size)

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

def preprocess_masks(mask_dataset, num_classes=5):
    """Konversi warna RGB pada masking menjadi label numerik & One-Hot Encoding."""
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
        # Convert one-hot encoded mask ke single channel mask
        mask = np.argmax(mask, axis=-1).astype(np.uint8)
    
    additional_targets = {'validity_mask': 'mask'} if validity_mask is not None else {}
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Resize(image_patch_size, image_patch_size)
    ], additional_targets=additional_targets)

    # Augmentasi dengan validity_mask
    if validity_mask is not None:
        augmented = transform(image=image, mask=mask, validity_mask=validity_mask)
        return augmented['image'], augmented['mask'], augmented['validity_mask']
    else:
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

def data_generator(images, masks, batch_size, validity_masks=None):
    while True:
        idx = np.random.permutation(len(images))
        for i in range(0, len(images), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_images = []
            batch_masks = []

            for j in batch_idx:
                if validity_masks is not None:
                    img, mask, _ = augment_data(images[j], masks[j], validity_masks[j])
                else:
                    img, mask = augment_data(images[j], masks[j])

                batch_images.append(img)
                
                # Kembalikan ke format one-hot encoding
                if mask.ndim == 2:
                    mask = to_categorical(mask, num_classes=num_classes)
                batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)

def model_unet(classes=num_classes):
    return multi_unet_model(n_classes=classes,image_height=image_patch_size, image_width=image_patch_size, image_channels=3)

def plot_training_results(history, epochs):
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

    filename = f"report/training_results_{epochs}e_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafik hasil pelatihan disimpan sebagai {filename}")

    plt.show()

def visualize_sample_data(X, y, validity_mask=None, num_samples=5):
    indices = np.random.choice(len(X), num_samples, replace=False)

    plt.figure(figsize=(15, num_samples * 3))
    for i, idx in enumerate(indices):
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(X[idx])
        plt.title(f"Sample {idx} - Image")
        plt.axis('off')

        plt.subplot(num_samples, 3, i*3 + 2)
        masks_display = np.argmax(y[idx], axis=-1)
        plt.imshow(masks_display, cmap='viridis')
        plt.title(f"Sample {idx} - Masks")
        plt.axis('off')

        if validity_mask is not None:
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(validity_mask[idx], cmap='gray')
            plt.title(f"Sample {idx} - Validity masks")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"report/visualisation_sample_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()

def plot_augmentation(images, masks, validity_masks=None):
    # Pastikan mask dan validity_mask memiliki dimensi yang sesuai
    # Konversi mask ke format yang sesuai jika diperlukan
    if masks.ndim == 3 and masks.shape[-1] > 1:
        # One-hot encoded mask, konversi ke single channel untuk augmentasi
        mask_single_channel = np.argmax(masks, axis=-1).astype(np.uint8)
    else:
        mask_single_channel = masks.astype(np.uint8)
    
    augmentations = [
        ("Flip Horizontal", A.HorizontalFlip(p=1.0)),
        ("Flip Vertical", A.VerticalFlip(p=1.0)),
        ("Rotation", A.Rotate(limit=45, p=1.0)),
        ("Brightness Contrast", A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)),
        ("Elastic Transform", A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0)),
        ("Random Scale", A.RandomScale(scale_limit=0.3, p=1.0)),
    ]

    plt.figure(figsize=(18, len(augmentations) * 3))

    for i, (name, aug) in enumerate(augmentations):
        additional_targets = {'validity_mask': 'mask'} if validity_masks is not None else {}
        
        transform = A.Compose(
            [aug, A.Resize(image_patch_size, image_patch_size)],
            additional_targets=additional_targets
        )
        
        if validity_masks is not None:
            augmented = transform(image=images, mask=mask_single_channel, validity_mask=validity_masks)
            aug_image, aug_mask, aug_valid = augmented['image'], augmented['mask'], augmented['validity_mask']
        else:
            augmented = transform(image=images, mask=mask_single_channel)
            aug_image, aug_mask = augmented['image'], augmented['mask']
            aug_valid = None
        
        plt.subplot(len(augmentations), 3 if validity_masks is not None else 2, i*3 + 1 if validity_masks is not None else i*2 + 1)
        plt.imshow(aug_image)
        plt.title(f"{name} - Gambar")
        plt.axis('off')

        plt.subplot(len(augmentations), 3 if validity_masks is not None else 2, i*3 + 2 if validity_masks is not None else i*2 + 2)
        plt.imshow(aug_mask, cmap='viridis')
        plt.title(f"{name} - Masking")
        plt.axis('off')

        if validity_masks is not None:
            plt.subplot(len(augmentations), 3, i*3 + 3)
            plt.imshow(aug_valid, cmap='gray')
            plt.title(f"{name} - Validity masks")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"report/augmentasi_sample_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()

def show_augmentation(image_dataset, mask_dataset, validity_masks=None, num_samples=1):
    indices = np.random.choice(len(image_dataset), min(num_samples, len(image_dataset)), replace=False)
    
    for idx in indices:
        image = image_dataset[idx]
        mask = mask_dataset[idx]
        validity_mask = validity_masks[idx] if validity_masks is not None else None
        plot_augmentation(image, mask, validity_mask)

def visualize_predict(model, X_test, y_test, num_sample=5):
    indices = np.random.choice(len(X_test), num_sample, replace=False)

    plt.figure(figsize=(15, num_sample * 3))

    for i, idx in enumerate(indices):
        plt.subplot(num_sample, 3, i*3 + 1)
        plt.imshow(X_test[idx])
        plt.title(f"Sample {idx} - Input")
        plt.axis('off')

        plt.subplot(num_sample, 3, i*3 + 2)
        ground_truth = np.argmax(y_test[idx], axis=-1)
        plt.imshow(ground_truth, cmap='viridis')
        plt.title(f"Sample {idx} - ground truth")
        plt.axis('off')

        plt.subplot(num_sample, 3, i*3 + 3)
        input_img = np.expand_dims(X_test[idx], axis=0)
        pred_mask = model.predict(input_img)
        pred_mask = np.argmax(pred_mask[0], axis=-1)
        plt.imshow(pred_mask, cmap='viridis')
        plt.title(f"Sample {idx} - Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"report/prediksi_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=300)
    plt.show()

def train_model(epochs, batch_size, valid_threshold=0.1):
    # Create necessary directories
    os.makedirs('report', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Load dataset
    image_dataset, mask_dataset, validity_masks = load_dataset(valid_threshold)
    
    # Preprocess masks
    labels = preprocess_masks(mask_dataset, num_classes=num_classes)
    
    print("Menampilkan sample data")
    visualize_sample_data(image_dataset, labels, validity_masks, num_samples=3)
    
    print("Menampilkan augmentasi data")
    try:
        # Gunakan sampel pertama saja untuk menampilkan augmentasi
        sample_idx = 0
        show_augmentation(
            image_dataset[sample_idx:sample_idx+1], 
            labels[sample_idx:sample_idx+1], 
            validity_masks[sample_idx:sample_idx+1] if validity_masks is not None else None, 
            num_samples=1
        )
    except Exception as e:
        print(f"Error saat menampilkan augmentasi: {e}")
        # Lanjutkan proses training meskipun visualisasi augmentasi gagal

    metrics = ['accuracy', jaccard_coef]

    weights = np.array([1.0, 0.5, 1.5, 1.0, 1.0])

    dice_loss = sm.losses.DiceLoss(class_weights=weights)
    focal_loss = sm.losses.CategoricalFocalLoss(weights)
    total_loss = dice_loss + focal_loss

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels, test_size=0.2, random_state=42)

    model = model_unet(classes=num_classes)
    model.compile(optimizer=Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7), loss=total_loss, metrics=metrics)

    best_model_filename = f"model/best_model_{epochs}e_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"
    final_model_filename = f"model/HGU_model_{epochs}e_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.h5"

    checkpoint = ModelCheckpoint(best_model_filename, monitor='val_loss', save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6)

    model.summary()

    history = model.fit(
        data_generator(X_train, y_train, batch_size, validity_masks),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_test, y_test),
        epochs=epochs,
        callbacks=[checkpoint, reduce_lr]
    )

    model.save(final_model_filename)
    print(f"Model akhir disimpan sebagai {final_model_filename}")
    plot_training_results(history, epochs)
    
    test_result = model.evaluate(X_test, y_test)
    print(f"Loss pada data testing: {test_result[0]}")
    print(f"Accuracy pada data testing: {test_result[1]}")
    print(f"MIoU pada data testing: {test_result[2]}")

    visualize_predict(model, X_test, y_test, num_sample=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    train_model(args.epochs, args.batch_size)