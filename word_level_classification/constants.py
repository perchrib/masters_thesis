import yaml

# Load yaml configuration file
with open('../config/config_word.yaml', 'r') as f:
    config_file = yaml.load(f)

# Prediction type
PREDICTION_TYPE = config_file['PREDICTION_TYPE']
GENDER = 'gender'
AGE = 'age'

# Directories
TEXT_DATA_DIR = '../data/txt/'
EMBEDDINGS_NATIVE_DIR = '../embeddings_native/'
EMBEDDINGS_INDEX_DIR = '../embeddings_index/'
LOG_DIR = '../logs/'

# Text pre-processing
MAX_NB_WORDS = config_file['pre-processing']['MAX_NB_WORDS']
MAX_SEQUENCE_LENGTH = config_file['pre-processing']['MAX_SEQUENCE_LENGTH']
VALIDATION_SPLIT = config_file['pre-processing']['VALIDATION_SPLIT']
EMBEDDINGS_INDEX = config_file['pre-processing']['EMBEDDINGS_INDEX']

# Model
MODEL_NAME = config_file['model']['name']
MODEL_OPTIMIZER = config_file['model']['optimizer']
MODEL_LOSS = config_file['model']['loss']
MODEL_METRICS = config_file['model']['metrics']
NB_EPOCHS = config_file['model']['nb-epochs']
BATCH_SIZE = config_file['model']['batch-size']