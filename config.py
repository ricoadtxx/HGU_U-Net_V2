import datetime

DATA_DIR = "dataset_new/"
DATA_NAME = "DBX"
IMAGE_PATCH_SIZE = 128
NUM_CLASSES = 5
VALID_THRESHOLD = 0.1
THRESHOLD = 2

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
CLASS_WEIGHTS = [0.5, 1.0, 2.0, 1.5, 1.0]

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BEST_MODEL_PATH = f"model/best_model_{TIMESTAMP}.h5"
FINAL_MODEL_PATH = f"model/final_model_{TIMESTAMP}.h5"

CLASS_NAMES = ['ground', 'hutan', 'palmoil', 'urban', 'vegetation']