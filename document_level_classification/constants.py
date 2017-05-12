
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
DIM_REDUCTION_SIZE = 200



# Model
MODEL_OPTIMIZER = 'adam'

# Standard!
MODEL_LOSS = 'categorical_crossentropy'

# For Logistic Regression
Log_Reg = False
#MODEL_LOSS = 'binary_crossentropy'

MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 128
