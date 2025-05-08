import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from scipy.spatial import cKDTree
from PIL import Image
import argparse

# =====================
# load CRS input tif
# =====================

def load_metadata(input_tif):
    with rasterio.open(input_tif) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        pixel_size_x = abs(transform[0])  # Resolusi X dalam meter/piksel
        pixel_size_y = abs(transform[4])  # Resolusi Y dalam meter/piksel
        pixel_area = pixel_size_x * pixel_size_y  # Luas 1 piksel dalam meter persegi
    return meta, transform, crs, pixel_area

# =====================
# load image hasil PNG
# =====================

def load_segmentation_image(input_png):
    image = Image.open(input_png).convert("RGB")
    return np.array(image)

# =====================
# dekoding kelas label
# =====================

def decode_segmentation(segmentasi_array, decoded_classes):
    height, width, _ = segmentasi_array.shape
    class_array = np.zeros((height, width), dtype=np.uint8)
    for color, class_id in decoded_classes.items():
        mask = np.all(segmentasi_array == color, axis=-1)
        class_array[mask] = class_id
    return class_array

# =====================
# menyimpan tif
# =====================

def save_geotiff(output_tif, class_array, meta, nodata_value):
    meta.update({"dtype": rasterio.uint8, "nodata": nodata_value, "count": 1})
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(class_array, 1)
    print(f"GeoTIFF berhasil disimpan: {output_tif}")

def generate_polygons(class_array, transform, class_labels):
    shapes_gen = shapes(class_array, transform=transform)
    polygons, values, labels = [], [], []
    for geom, value in shapes_gen:
        poly = shape(geom).buffer(1).buffer(-1)  
        polygons.append(poly)
        values.append(value)
        labels.append(class_labels.get(value, "Unknown"))
    return polygons, values, labels

def create_geodataframe(polygons, values, labels, crs):
    return gpd.GeoDataFrame({"class": values, "label": labels, "geometry": polygons}, crs=crs)

def interpolate_small_polygons(gdf, min_area_threshold=200):
    small_polygons = gdf[gdf.geometry.area < min_area_threshold]
    large_polygons = gdf[gdf.geometry.area >= min_area_threshold]
    if not small_polygons.empty and not large_polygons.empty:
        small_centroids = np.array([geom.centroid.coords[0] for geom in small_polygons.geometry])
        large_centroids = np.array([geom.centroid.coords[0] for geom in large_polygons.geometry])
        tree = cKDTree(large_centroids)
        _, indices = tree.query(small_centroids)
        for small_idx, large_idx in zip(small_polygons.index, indices):
            gdf.at[small_idx, "class"] = large_polygons.iloc[large_idx]["class"]
            gdf.at[small_idx, "label"] = large_polygons.iloc[large_idx]["label"]
    return gdf

# =====================
# hitung luas per kelas
# =====================

def calculate_area(gdf, class_array, pixel_area, class_labels):
    # Hitung luas berdasarkan geometri polygon
    gdf["luas_geom"] = gdf.geometry.area
    
    # Hitung jumlah piksel per kelas
    unique, counts = np.unique(class_array, return_counts=True)
    pixel_count_per_class = dict(zip(unique, counts))
    luas_pixel_based = {class_labels.get(cls, "Unknown"): count * pixel_area for cls, count in pixel_count_per_class.items()}
    
    print("\nTotal Luas Tiap Kelas (berdasarkan jumlah piksel):")
    for label, area in luas_pixel_based.items():
        print(f" - {label}: {area / 10000:.2f} hektare")
    
    return luas_pixel_based

# =====================
# Simpan SHP
# =====================

def save_shapefile(gdf, output_shp):
    gdf.to_file(output_shp)
    print(f"Shapefile berhasil disimpan: {output_shp}")

def main(input_tif, input_png, output_tif, output_shp, decoded_classes, class_labels, nodata_value=255):
    meta, transform, crs, pixel_area = load_metadata(input_tif)
    segmentasi_array = load_segmentation_image(input_png)
    class_array = decode_segmentation(segmentasi_array, decoded_classes)
    save_geotiff(output_tif, class_array, meta, nodata_value)
    
    polygons, values, labels = generate_polygons(class_array, transform, class_labels)
    gdf = create_geodataframe(polygons, values, labels, crs)
    gdf = interpolate_small_polygons(gdf)
    
    print("Menghitung luas berdasarkan jumlah piksel...")
    calculate_area(gdf, class_array, pixel_area, class_labels)
    
    print("Menyimpan data ke format Shapefile...")
    save_shapefile(gdf, output_shp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tif", type=str, required=True,) # untuk acuan koordinat
    parser.add_argument("--input_png", type=str, required=True,) # untuk acuan warna kelas segmentasi
    parser.add_argument("--output_tif", type=str, required=True,)
    parser.add_argument("--output_shp", type=str, required=True,)

    args = parser.parse_args()
    
    decoded_classes = {
        (167, 168, 167): 0, 
        (21, 194, 59): 1,
        (46, 15, 15): 2,
        (237, 92, 14): 3,
        (102, 237, 69): 4
    }
    class_labels = {
        0: 'ground',
        1: 'hutan',
        2: 'palmoil',
        3: 'urban',
        4: 'vegetation'
    }

    main(args.input_tif, args.input_png, args.output_tif, args.output_shp, decoded_classes, class_labels)
