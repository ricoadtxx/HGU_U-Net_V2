import os
import numpy as np
from collections import Counter
from PIL import Image

# Mapping warna → kelas
class_colors = {
    'ground': [167, 168, 167],
    'hutan': [21, 194, 59],
    'palmoil': [46, 15, 15],
    'urban': [237, 92, 14],
    'vegetation': [102, 237, 69]
}
# Buat reverse map: RGB tuple → nama kelas
color_to_class = {tuple(v): k for k, v in class_colors.items()}

def analyze_rgb_class_pixel_distribution(root_dir):
    class_pixel_counts = Counter()

    for tile_name in os.listdir(root_dir):
        tile_path = os.path.join(root_dir, tile_name)
        mask_dir = os.path.join(tile_path, "masking")

        if not os.path.isdir(mask_dir):
            continue

        for fname in os.listdir(mask_dir):
            if fname.endswith(".png"):
                path = os.path.join(mask_dir, fname)
                mask = Image.open(path).convert("RGB")
                mask_np = np.array(mask)
                h, w, _ = mask_np.shape

                # Ubah shape ke (H*W, 3)
                flat_pixels = mask_np.reshape(-1, 3)

                # Hitung semua warna unik
                unique_colors, counts = np.unique(flat_pixels, axis=0, return_counts=True)

                for color, count in zip(unique_colors, counts):
                    color_tuple = tuple(color)
                    class_name = color_to_class.get(color_tuple, None)

                    if class_name:
                        class_pixel_counts[class_name] += count
                    else:
                        class_pixel_counts["Unknown"] += count  # warnai yg tidak dikenali

    # Tampilkan hasil akhir
    print("Total pixel count per class:")
    for cls in class_colors.keys():
        print(f"  {cls}: {class_pixel_counts[cls]:,} pixels")
    if "Unknown" in class_pixel_counts:
        print(f"  Unknown (unmapped colors): {class_pixel_counts['Unknown']:,} pixels")

# Jalankan fungsi
label_root_folder = "dataset_new/DBX"
analyze_rgb_class_pixel_distribution(label_root_folder)
