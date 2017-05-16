import os
# Prediction type
GENDER = 'gender'
AGE = 'age'
PREDICTION_TYPE = GENDER

# Log directory
LOGS_DIR = '../logs/word_embedding_classification/'
MODEL_DIR = '../models/word_embedding_classification/'
WORD_INDEX_DIR = os.path.join(MODEL_DIR, 'word_index')

# Text pre-processing
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 15
EMBEDDINGS_INDEX = 'glove.twitter.27B.200d'

# Model
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 128