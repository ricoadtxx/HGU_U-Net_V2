import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def apply_clahe(image, clip_limit=4.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def percentile_stretch(image, lower_percentile=2, upper_percentile=98):
    p_low, p_high = np.percentile(image, (lower_percentile, upper_percentile))

    if p_high > p_low:
        stretched = np.clip((image - p_low) / (p_high - p_low) * 255, 0, 255)
    else:
        stretched = np.zeros_like(image, dtype=np.uint8)

    return stretched.astype(np.uint8)

def save_tiled_images(image_array, profile, tile_width=1000, tile_height=500, output_dir="tif/tiles/"):
    """ Membagi citra menjadi tile 1000x500 dan menyimpannya """
    os.makedirs(output_dir, exist_ok=True)
    
    height, width = image_array.shape[1], image_array.shape[2]
    tile_idx = 0

    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            tile = image_array[:, y:y+tile_height, x:x+tile_width]

            # Pastikan tile memiliki ukuran tetap 1000x500 sebelum menyimpan
            if tile.shape[1] == tile_height and tile.shape[2] == tile_width:
                tile_filename = os.path.join(output_dir, f"tile_{tile_idx}.tif")
                
                # Update metadata
                profile.update(width=tile_width, height=tile_height, dtype=rasterio.uint8)
                
                with rasterio.open(tile_filename, "w", **profile) as dst:
                    dst.write(tile)

                print(f"Tile {tile_idx} disimpan: {tile_filename}")
                tile_idx += 1

    print(f"Total tile disimpan: {tile_idx}")

# Path input dan output
input_path = "tif/data/hasil.tif"
output_path = "tif/pre-process/2.tif"

# Buka file TIFF
with rasterio.open(input_path) as dataset:
    array_piksel = dataset.read()
    profile = dataset.profile

print("Nilai piksel sebelum stretching:\n", array_piksel[:, :5, :5])

if array_piksel.shape[0] >= 3:
    original_rgb = np.stack([array_piksel[0], array_piksel[1], array_piksel[2]], axis=-1)
    original_rgb = ((original_rgb - original_rgb.min()) / (original_rgb.max() - original_rgb.min()) * 255).astype(np.uint8)
else:
    print("Gambar tidak memiliki cukup band untuk RGB!")

# Normalisasi tiap band
normalized_bands = np.zeros_like(array_piksel, dtype=np.uint8)

for i in range(array_piksel.shape[0]):
    band = array_piksel[i].astype(np.float32)
    normalized_bands[i] = percentile_stretch(band, lower_percentile=2, upper_percentile=98)

for i in range(normalized_bands.shape[0]):
    normalized_bands[i] = apply_clahe(normalized_bands[i], clip_limit=4.0, grid_size=(8, 8))

print("\n Nilai piksel setelah stretching:\n", normalized_bands[:, :5, :5])

if normalized_bands.shape[0] >= 3:
    band_red = normalized_bands[0]
    band_green = normalized_bands[1]
    band_blue = normalized_bands[2]

    stretched_rgb = np.stack([band_red, band_green, band_blue], axis=-1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Sebelum Stretching Kontras (RGB Asli)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(stretched_rgb)
    plt.title("Setelah Stretching Kontras")
    plt.axis("off")

    plt.show()
else:
    print("Tidak memiliki cukup band untuk RGB.")

# Simpan hasil pre-processing sebagai TIFF
profile.update(dtype=rasterio.uint8)

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(normalized_bands)

print(f"Hasil pre-processing disimpan sebagai: {output_path}")

# Panggil fungsi untuk menyimpan tile
save_tiled_images(normalized_bands, profile)
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def apply_clahe(image, clip_limit=4.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def percentile_stretch(image, lower_percentile=2, upper_percentile=98):
    p_low, p_high = np.percentile(image, (lower_percentile, upper_percentile))

    if p_high > p_low:
        stretched = np.clip((image - p_low) / (p_high - p_low) * 255, 0, 255)
    else:
        stretched = np.zeros_like(image, dtype=np.uint8)

    return stretched.astype(np.uint8)

def save_tiled_images(image_array, profile, tile_width=1000, tile_height=500, output_dir="tif/pre-process/tiles/"):
    """ Membagi citra menjadi tile 1000x500 dan menyimpannya """
    os.makedirs(output_dir, exist_ok=True)
    
    height, width = image_array.shape[1], image_array.shape[2]
    tile_idx = 0

    for y in range(0, height, tile_height):
        for x in range(0, width, tile_width):
            tile = image_array[:, y:y+tile_height, x:x+tile_width]

            # Pastikan tile memiliki ukuran tetap 1000x500 sebelum menyimpan
            if tile.shape[1] == tile_height and tile.shape[2] == tile_width:
                tile_filename = os.path.join(output_dir, f"tile_{tile_idx}.tif")
                
                # Update metadata
                profile.update(width=tile_width, height=tile_height, dtype=rasterio.uint8)
                
                with rasterio.open(tile_filename, "w", **profile) as dst:
                    dst.write(tile)

                print(f"Tile {tile_idx} disimpan: {tile_filename}")
                tile_idx += 1

    print(f"Total tile disimpan: {tile_idx}")

# Path input dan output
input_path = "tif/data/hasil.tif"
output_path = "tif/pre-process/2.tif"

# Buka file TIFF
with rasterio.open(input_path) as dataset:
    array_piksel = dataset.read()
    profile = dataset.profile
    
print("Nilai piksel sebelum stretching:\n", array_piksel[:, :5, :5])

if array_piksel.shape[0] >= 3:
    original_rgb = np.stack([array_piksel[0], array_piksel[1], array_piksel[2]], axis=-1)
    original_rgb = ((original_rgb - original_rgb.min()) / (original_rgb.max() - original_rgb.min()) * 255).astype(np.uint8)
else:
    print("Gambar tidak memiliki cukup band untuk RGB!")

# Normalisasi tiap band
normalized_bands = np.zeros_like(array_piksel, dtype=np.uint8)

for i in range(array_piksel.shape[0]):
    band = array_piksel[i].astype(np.float32)
    normalized_bands[i] = percentile_stretch(band, lower_percentile=2, upper_percentile=98)

for i in range(normalized_bands.shape[0]):
    normalized_bands[i] = apply_clahe(normalized_bands[i], clip_limit=4.0, grid_size=(8, 8))

print("\n Nilai piksel setelah stretching:\n", normalized_bands[:, :5, :5])

if normalized_bands.shape[0] >= 3:
    band_red = normalized_bands[0]
    band_green = normalized_bands[1]
    band_blue = normalized_bands[2]

    stretched_rgb = np.stack([band_red, band_green, band_blue], axis=-1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Sebelum Stretching Kontras (RGB Asli)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(stretched_rgb)
    plt.title("Setelah Stretching Kontras")
    plt.axis("off")

    plt.show()
else:
    print("Tidak memiliki cukup band untuk RGB.")

# Simpan hasil pre-processing sebagai TIFF
profile.update(dtype=rasterio.uint8)

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(normalized_bands)

print(f"Hasil pre-processing disimpan sebagai: {output_path}")

# Panggil fungsi untuk menyimpan tile
save_tiled_images(normalized_bands, profile)
