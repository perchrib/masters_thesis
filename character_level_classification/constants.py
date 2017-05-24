import os
from helpers.global_constants import REM_STOPWORDS, LEMMATIZE, REM_EMOTICONS, REM_PUNCTUATION, REM_INTERNET_TERMS, LOWERCASE
# Prediction type
GENDER = 'gender'
AGE = 'age'
PREDICTION_TYPE = GENDER


# Log directory
LOGS_DIR = '../logs/character_level_classification'
MODEL_DIR = '../models/character_level_classification'
CHAR_INDEX_DIR = os.path.join(MODEL_DIR, 'char_index')

# Text pre-processing
MAX_SEQUENCE_LENGTH = 100


# Filtering constants
FILTERS = {
    REM_STOPWORDS: True,
    LEMMATIZE: False,
    REM_PUNCTUATION: False,
    REM_EMOTICONS: False,
    LOWERCASE: True
}

# Model
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 128
