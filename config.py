import datetime
from class_weights import calculate_class_weights

dataset_dir = "dataset_new/DBX"  # Ganti dengan path dataset kamu

DATA_DIR = "dataset_new/"
DATA_NAME = "DBX"
IMAGE_PATCH_SIZE = 256
NUM_CLASSES = 5
VALID_THRESHOLD = 0.1
THRESHOLD = 1

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0005
CLASS_WEIGHTS = calculate_class_weights(dataset_dir)

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BEST_MODEL_PATH = f"model/best_model_{TIMESTAMP}.h5"
FINAL_MODEL_PATH = f"model/final_model_{TIMESTAMP}.h5"

CLASS_NAMES = ['ground', 'hutan', 'palmoil', 'urban', 'vegetation']