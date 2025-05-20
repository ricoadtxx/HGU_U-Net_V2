import os
import numpy as np
from PIL import Image

# Mapping RGB values ke nama kelas
class_colors = {
    'ground': [167, 168, 167],
    'hutan': [21, 194, 59],
    'palmoil': [46, 15, 15],
    'urban': [237, 92, 14],
    'vegetation': [102, 237, 69]
}

# Fungsi untuk menghitung total piksel per kelas
def calculate_class_weights(dataset_dir):
    pixel_counts = {cls: 0 for cls in class_colors}  # Inisialisasi hitungan per kelas

    for tile_name in os.listdir(dataset_dir):
        tile_path = os.path.join(dataset_dir, tile_name)
        mask_dir = os.path.join(tile_path, "masking")

        if not os.path.isdir(mask_dir):
            continue

        # Iterasi setiap mask di folder masks
        for fname in os.listdir(mask_dir):
            if fname.endswith(".png"):
                mask_path = os.path.join(mask_dir, fname)
                mask = Image.open(mask_path).convert("RGB")
                mask_np = np.array(mask)

                # Iterasi setiap piksel untuk menghitung jumlah tiap kelas
                for class_name, color in class_colors.items():  # Di sini kita ambil class_name dan RGB color langsung
                    # Tidak perlu mengonversi menjadi tuple, langsung gunakan list
                    count = np.sum(np.all(mask_np == color, axis=-1))  # Hitung jumlah piksel warna ini
                    pixel_counts[class_name] += count

    # Hitung total piksel
    total_pixels = sum(pixel_counts.values())

    # Hitung class weights sebagai invers frekuensi
    class_weights = {cls: total_pixels / count for cls, count in pixel_counts.items() if count > 0}

    # Normalisasi bobot kelas
    max_weight = max(class_weights.values())
    normalized_weights = {cls: class_weights[cls] / max_weight for cls in class_weights}

    print(f"Class Weights (normalized): {normalized_weights}")
    return normalized_weights

