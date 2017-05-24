import os
from helpers.global_constants import REM_STOPWORDS, LEMMATIZE, REM_EMOTICONS, REM_PUNCTUATION, REM_INTERNET_TERMS
# Prediction type
GENDER = 'gender'
AGE = 'age'
PREDICTION_TYPE = GENDER

# Log directory
LOGS_DIR = '../logs/word_embedding_classification/'
MODEL_DIR = '../models/word_embedding_classification/'
WORD_INDEX_DIR = os.path.join(MODEL_DIR, 'word_index')

# Text pre-processing
MAX_NB_WORDS = 5000   # TODO:
MAX_SEQUENCE_LENGTH = 15
EMBEDDINGS_INDEX = 'glove.twitter.27B.200d'

# Text filtering constants
FILTERS = {
    REM_STOPWORDS: True,
    LEMMATIZE: False,
    REM_PUNCTUATION: False,
    REM_EMOTICONS: False
}

# Model
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRICS = ['accuracy']
NB_EPOCHS = 100
BATCH_SIZE = 128