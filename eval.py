import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
import argparse
import os

def normalize_mask(mask, num_classes=5):
    """Konversi mask dari rentang 0-255 ke 0-(num_classes-1) jika perlu."""
    max_val = mask.max()
    
    # Jika mask memiliki nilai lebih dari jumlah kelas, normalisasi
    if max_val > num_classes - 1:
        print(f"Normalisasi mask dari rentang [0, {max_val}] ke [0, {num_classes-1}]")
        mask = (mask / (255 / (num_classes - 1))).astype(np.uint8)
    
    return mask

def compute_miou(pred_mask, gt_mask, num_classes=5):
    """Menghitung MIoU antara prediksi dan ground truth."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Ukuran prediksi {pred_mask.shape} tidak cocok dengan GT {gt_mask.shape}")

    # Normalisasi mask jika perlu
    pred_mask = normalize_mask(pred_mask, num_classes)
    gt_mask = normalize_mask(gt_mask, num_classes)

    # Cek nilai maksimum setelah normalisasi
    print(f"Nilai Maksimum Setelah Normalisasi - Prediksi: {pred_mask.max()}, GT: {gt_mask.max()}")

    # Buat objek MeanIoU
    miou_metric = MeanIoU(num_classes=num_classes)
    miou_metric.update_state(gt_mask.flatten(), pred_mask.flatten())

    miou_value = miou_metric.result().numpy()
    print(f"MIoU: {miou_value:.4f}")

    return miou_value

def visualize_results(image_path, gt_path, pred_path, output_path):
    """Menampilkan gambar asli, ground truth, dan hasil prediksi berdampingan serta menyimpannya."""
    # Load gambar asli
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load ground truth
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # Load hasil prediksi
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if image is None or gt_mask is None or pred_mask is None:
        raise ValueError("Pastikan semua file gambar tersedia.")

    # Hitung MIoU
    miou = compute_miou(pred_mask, gt_mask)

    # Tampilkan gambar
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title("Gambar Asli")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap="gray")
    axs[1].set_title("Ground Truth Masking")
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap="gray")
    axs[2].set_title(f"Prediksi (MIoU: {miou:.4f})")
    axs[2].axis("off")

    plt.tight_layout()

    # Simpan hasil visualisasi ke file
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Hasil disimpan sebagai {output_path}")

    # Tampilkan hasil
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path ke gambar asli")
    parser.add_argument('--gt', type=str, required=True, help="Path ke ground truth")
    parser.add_argument('--pred', type=str, required=True, help="Path ke hasil prediksi")
    parser.add_argument('--output', type=str, required=True, help="Path untuk menyimpan hasil evaluasi")
    
    args = parser.parse_args()

    visualize_results(args.image, args.gt, args.pred, args.output)
