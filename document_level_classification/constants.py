
# Prediction type
GENDER = 'gender'
AGE = 'age'
PREDICTION_TYPE = GENDER


# Log directory
LOGS_DIR = '../logs/document_level_classification'
MODEL_DIR = '../models/document_level_classification'

# Text pre-processing
MAX_FEATURE_LENGTH = 10000
N_GRAM = (1, 1)
DIM_REDUCTION = True



# Model
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 128
