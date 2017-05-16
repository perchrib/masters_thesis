from helpers.global_constants import REM_STOPWORDS, LEMMATIZE, REM_EMOTICONS, REM_PUNCTUATION
# Prediction type
GENDER = 'gender'
AGE = 'age'
C_BAG_OF_WORDS = 'bow'
C_TF_IDF = 'tfidf'

PREDICTION_TYPE = GENDER


# Log directory
LOGS_DIR = '../logs/document_level_classification'
MODEL_DIR = '../models/document_level_classification'

# Autoencoder
DIM_REDUCTION = False
DIM_REDUCTION_SIZE = 50

if not DIM_REDUCTION:
    DIM_REDUCTION_SIZE = None

#############################################
# Model (TUNING PARAMETERS HERE!)
MODEL_TYPE = "base" # set TYPE = "" when not using "base"
LAYERS = [512]
# Regularization
DROPOUT = 0
L1 = 0
L2 = 0

# Text pre-processing
FEATURE_MODEL = C_BAG_OF_WORDS
MAX_FEATURE_LENGTH = 10000
N_GRAM = (1, 1)

###############################################
# Standard!
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'softmax'
CATEGORICAL = True

# For Logistic Regression
Log_Reg = False
if Log_Reg:
    CATEGORICAL = False
    MODEL_LOSS = 'binary_crossentropy'

MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 128


# Filtering constants
FILTERS = {
    REM_STOPWORDS: True,
    LEMMATIZE: False,
    REM_PUNCTUATION: False,
    REM_EMOTICONS: False
}
