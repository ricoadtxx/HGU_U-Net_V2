import os
import rasterio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import load_model
from tqdm import tqdm
from losses import jaccard_coef
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd
from scipy.ndimage import generic_filter
from model import (attention_gate, spatial_attention_module, 
                          residual_block)

# Konfigurasi awal
os.environ["SM_FRAMEWORK"] = "tf.keras"
image_patch_size = 256
num_classes = 5

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def create_nodata_mask(image, threshold=2):
    no_data_mask = ~np.all(image < threshold, axis=-1)
    return no_data_mask.astype(np.uint8)

def is_valid_patch(patch, valid_threshold=0.1):
    no_data_mask = np.all(patch < 10, axis=-1)
    valid_ratio = 1.0 - (np.sum(no_data_mask) / no_data_mask.size)
    return valid_ratio >= valid_threshold

def load_custom_model(model_path):
    dependencies = {
        'jaccard_coef': jaccard_coef,
        'attention_gate': attention_gate,
        'spatial_attention_module': spatial_attention_module,
        'residual_block': residual_block
    }
    model = load_model(model_path, custom_objects=dependencies, compile=False)
    return model

# def mode_filter(label_map, size=5):
#     h, w = label_map.shape
#     result = np.zeros_like(label_map)
    
#     pad_size = size // 2
#     padded = np.pad(label_map, pad_size, mode='reflect')
    
#     with tqdm(total=h*w, desc="Smoothing", unit="piksel") as progress_bar:
#         for i in range(h):
#             for j in range(w):
#                 window = padded[i:i+size, j:j+size].flatten()
#                 vals, counts = np.unique(window, return_counts=True)
#                 result[i, j] = vals[np.argmax(counts)]
#                 progress_bar.update(1)
    
#     return result

# def label_to_rgb(label_map, color_list):
#     rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
#     for idx, color in enumerate(color_list):
#         rgb[label_map == idx] = color
#     return rgb

def predict_large_image(model, image_path, output_path=None, patch_size=image_patch_size, overlap=128, valid_threshold=0.1):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")

    image_rgb = image
    h, w = image_rgb.shape[:2]
    step = patch_size - overlap

    pad_h = (h // step + (1 if h % step != 0 else 0)) * step + overlap - h
    pad_w = (w // step + (1 if w % step != 0 else 0)) * step + overlap - w

    padded_image = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_h, padded_w = padded_image.shape[:2]

    colors = {
        'ground': [167, 168, 167],
        'hutan': [21, 194, 59],
        'palmoil': [46, 15, 15],
        'urban': [237, 92, 14],
        'vegetation': [102, 237, 69]
    }
    color_list = list(colors.values())

    prediction = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    patch_coords = [(y, x) for y in range(0, padded_h - overlap, step) for x in range(0, padded_w - overlap, step)]

    for (y, x) in tqdm(patch_coords, desc="Memproses patch"):
        patch = padded_image[y:y+patch_size, x:x+patch_size]

        if is_valid_patch(patch, valid_threshold):
            patch_norm = normalize_image(patch)
            pred = model.predict(np.expand_dims(patch_norm, axis=0), verbose=0)[0]
            pred_labels = np.argmax(pred, axis=-1)
            pred_colored = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

            for i in range(num_classes):
                pred_colored[pred_labels == i] = color_list[i]

            prediction[y:y+patch_size, x:x+patch_size] += pred_colored
            count_map[y:y+patch_size, x:x+patch_size] += 1

    mask = count_map > 0
    for c in range(3):
        prediction[:, :, c] = np.divide(prediction[:, :, c], count_map, out=np.zeros_like(prediction[:, :, c]), where=mask)

    segmented_image = prediction[:h, :w].astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
        print(f"Hasil segmentasi disimpan ke {output_path}")

    return segmented_image

def rgb_to_label(segmented_rgb, color_list):
    label_map = np.zeros(segmented_rgb.shape[:2], dtype=np.uint8)
    for idx, color in enumerate(color_list):
        mask = np.all(segmented_rgb == color, axis=-1)
        label_map[mask] = idx
    return label_map

def save_geotiff(label_map, reference_image_path, output_path):
    with rasterio.open(reference_image_path) as src:
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': 'uint8'})
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(label_map, 1)

def save_shapefile_from_label(label_map, reference_image_path, shp_output_path, class_names):
    with rasterio.open(reference_image_path) as src:
        transform = src.transform
        crs = src.crs

    results = (
        {'properties': {'class_id': v}, 'geometry': s}
        for s, v in shapes(label_map, mask=label_map > 0, transform=transform)
    )
    geoms = list(results)
    if not geoms:
        print("Tidak ada geometri yang berhasil diekstrak.")
        return

    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
    gdf['class_name'] = gdf['class_id'].map(lambda x: class_names[int(x)] if not np.isnan(x) and int(x) < len(class_names) else "Unknown")
    gdf.to_file(shp_output_path)
    print(f"Shapefile disimpan ke {shp_output_path}")

def calculate_area_from_shapefile(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    if not gdf.crs.is_projected:
        print("\n⚠️ CRS saat ini bersifat geografis (derajat), luas tidak akurat.")
        print("Disarankan menggunakan CRS terproyeksi seperti UTM untuk hasil luas yang akurat.\n")
    gdf['area_ha'] = gdf.geometry.area / 10000.0
    area_by_class = gdf.groupby('class_name')['area_ha'].sum()

    print("\nLuas per kelas:")
    for cls, area in area_by_class.items():
        print(f"- {cls}: {area:.2f} hektar")

    return area_by_class

def visualize_results(input_image_path, segmented_image, class_names, color_list, save_path=None):
    input_img = rasterio.open(input_image_path).read([1, 2, 3])
    input_img = np.transpose(input_img, (1, 2, 0)).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    ax[0].imshow(input_img)
    ax[0].set_title("Citra Input")
    ax[0].axis("off")

    ax[1].imshow(segmented_image)
    ax[1].set_title("Hasil Segmentasi")
    ax[1].axis("off")

    patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None,
                        color=np.array(c)/255.0,
                        label="{:s}".format(name))[0]
               for c, name in zip(color_list, class_names)]
    fig.legend(handles=patches, loc='lower center', ncol=len(class_names), fontsize='large')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualisasi disimpan ke {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Prediksi segmentasi gambar menggunakan model U-Net')
    parser.add_argument('--model', type=str, required=True, help='Path ke model terlatih (.h5)')
    parser.add_argument('--input', type=str, required=True, help='Path ke gambar input')
    parser.add_argument('--output', type=str, default='prediction_result.png', help='Path untuk menyimpan hasil segmentasi')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold untuk validitas patch')
    args = parser.parse_args()

    print(f"Memuat model dari {args.model}...")
    model = load_custom_model(args.model)
    print("Model berhasil dimuat")

    segmented_image = predict_large_image(model, args.input, args.output, valid_threshold=args.threshold)

    color_list = [
        [167, 168, 167],  # ground
        [21, 194, 59],    # hutan
        [46, 15, 15],     # palmoil
        [237, 92, 14],    # urban
        [102, 237, 69],   # vegetation
    ]
    class_names = ['ground', 'hutan', 'palmoil', 'urban', 'vegetation']

    label_map = rgb_to_label(segmented_image, color_list)

    tif_output_path = os.path.splitext(args.output)[0] + '.tif'
    save_geotiff(label_map, args.input, tif_output_path)

    shp_output_path = os.path.splitext(args.output)[0] + '.shp'
    save_shapefile_from_label(label_map, args.input, shp_output_path, class_names)

    calculate_area_from_shapefile(shp_output_path)

    visualization_output = os.path.splitext(args.output)[0] + '_viz.png'
    visualize_results(args.input, segmented_image, class_names, color_list, save_path=visualization_output)

if __name__ == "__main__":
    main()
