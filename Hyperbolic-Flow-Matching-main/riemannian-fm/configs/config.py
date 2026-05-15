SPHERE_MODEL = "sphere-small-small-animal-faces-256px"
DATA_DIR = "../../sphere-encoder-main/workspace/experiments/" + SPHERE_MODEL + "/encoding/"

RAW_DATA_PATH = DATA_DIR + "encoded_dataset.npz"
PROC_DATA_PATH = DATA_DIR + "processed_dataset.npz"
OUTPUT_DATA_PATH = DATA_DIR + "output_encodings.npz"

SPHERE_DIMS = [256, 4]

SQUEEZE_DATA = False
SQUEEZE_ALPHA = 0.0
