from helpers.helper_functions import load_yaml

# Load yaml configuration file
config_file = load_yaml('../config/config_word.yaml')

# Prediction type
PREDICTION_TYPE = config_file['PREDICTION_TYPE']
GENDER = 'gender'
AGE = 'age'

# Log directory
LOGS_DIR = '../logs/word_level_classification'

# Text pre-processing
MAX_NB_WORDS = config_file['pre-processing']['MAX_NB_WORDS']
MAX_SEQUENCE_LENGTH = config_file['pre-processing']['MAX_SEQUENCE_LENGTH']
VALIDATION_SPLIT = config_file['pre-processing']['VALIDATION_SPLIT']
EMBEDDINGS_INDEX = config_file['pre-processing']['EMBEDDINGS_INDEX']

# Model
MODEL_OPTIMIZER = config_file['model']['optimizer']
MODEL_LOSS = config_file['model']['loss']
MODEL_METRICS = config_file['model']['metrics']
NB_EPOCHS = config_file['model']['nb-epochs']
BATCH_SIZE = config_file['model']['batch-size']