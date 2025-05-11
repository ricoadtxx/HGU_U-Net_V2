import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import segmentation_models as sm


from model import multi_unet_model
from losses import jaccard_coef
from processing import load_dataset, preprocess_masks, data_generator
from visualization import visualize_sample_data, plot_training_results, visualize_predict, show_augmentation

from config import (
    IMAGE_PATCH_SIZE, NUM_CLASSES, CLASS_WEIGHTS, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, VALID_THRESHOLD, CLASS_NAMES
)

def train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, valid_threshold=VALID_THRESHOLD):
    os.makedirs('report', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    # Load and preprocess data
    image_dataset, mask_dataset, validity_masks = load_dataset(valid_threshold)
    labels = preprocess_masks(mask_dataset, num_classes=NUM_CLASSES)

    # Visualisasi data awal
    visualize_sample_data(image_dataset, labels, validity_mask=validity_masks, num_samples=3)
    show_augmentation(image_dataset, labels, validity_masks, num_samples=1)

    # Model dan loss
    model = multi_unet_model(n_classes=NUM_CLASSES, 
                            image_height=IMAGE_PATCH_SIZE, 
                            image_width=IMAGE_PATCH_SIZE,
                            image_channels=3)
    
    metrics = ['accuracy', jaccard_coef]
    
    # Handle case if segmentation_models not installed or has compatibility issues
    try:
        # Coba beberapa alternatif untuk menggunakan class_weights
        # Alternatif 1: Gunakan parameter yang tersedia di versi segmentation_models Anda
        try:
            dice_loss = sm.losses.DiceLoss()
            focal_loss = sm.losses.CategoricalFocalLoss()
            total_loss = dice_loss + focal_loss
            print("Using default DiceLoss + CategoricalFocalLoss")
        except Exception as e1:
            print(f"Failed to create combined loss: {e1}")
            
            # Alternatif 2: Coba gunakan binary_crossentropy + dice_loss
            try:
                total_loss = 'categorical_crossentropy'
                print("Using categorical_crossentropy as loss")
            except Exception as e2:
                print(f"Failed to set backup loss: {e2}")
                total_loss = 'categorical_crossentropy'
    except ImportError:
        print("Warning: segmentation_models not available, using categorical_crossentropy as loss")
        total_loss = 'categorical_crossentropy'

    model.compile(
        optimizer=Nadam(learning_rate=LEARNING_RATE),
        loss=total_loss,
        metrics=metrics
    )

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels, test_size=0.2, random_state=42)

    callbacks = [
        ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    model.summary()

    # Training
    history = model.fit(
        data_generator(X_train, y_train, batch_size, validity_masks),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_test, y_test),
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")

    # Plot hasil pelatihan
    plot_training_results(history, epochs)

    # Evaluasi
    evaluation = model.evaluate(X_test, y_test)
    
    # Handle multiple return values from evaluate
    if isinstance(evaluation, list):
        if len(evaluation) >= 3:
            loss, acc, miou = evaluation
        else:
            loss = evaluation[0]
            acc = evaluation[1] if len(evaluation) > 1 else None
            miou = None
    else:
        loss = evaluation
        acc = None
        miou = None
        
    print(f"Test Loss: {loss}")
    if acc is not None: print(f"Test Accuracy: {acc}")
    if miou is not None: print(f"Test mIoU: {miou}")

    # Visualisasi prediksi
    visualize_predict(model, X_test, y_test, num_samples=5)