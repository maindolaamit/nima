import os
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'nima', 'weights')
MODELS_JSON_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, 'nima', 'model', 'models.json')
AVA_DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'data', 'AVA')
AVA_DATASET_IMAGES_DIR = os.path.join(AVA_DATASET_DIR, 'images')
AVA_FILE = os.path.join(AVA_DATASET_DIR, 'AVA.txt')
AVA_CSV = os.path.join(AVA_DATASET_DIR, 'AVA.csv')
