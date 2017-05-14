
# Prediction type
GENDER = 'gender'
AGE = 'age'
C_BAG_OF_WORDS = 'bow'
C_TF_IDF = 'tfidf'

PREDICTION_TYPE = GENDER


# Log directory
LOGS_DIR = '../logs/document_level_classification'
MODEL_DIR = '../models/document_level_classification'

# Text pre-processing
FEATURE_MODEL = C_BAG_OF_WORDS
MAX_FEATURE_LENGTH = 10000
N_GRAM = (1, 1)

# Autoencoder
DIM_REDUCTION = False
DIM_REDUCTION_SIZE = 50



# Model
MODEL_OPTIMIZER = 'adam'

# Standard!
MODEL_LOSS = 'categorical_crossentropy'
CATEGORICAL = True

# For Logistic Regression
Log_Reg = False
if Log_Reg:
    CATEGORICAL = False
    MODEL_LOSS = 'binary_crossentropy'

MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 128
