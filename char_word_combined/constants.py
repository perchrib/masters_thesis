
# Prediction type
GENDER = 'gender'
AGE = 'age'
PREDICTION_TYPE = GENDER


# Log directory
LOGS_DIR = '../logs/char_word_combined'
MODEL_DIR = '../models/char_word_combined'

# Text pre-processing
MAX_NB_WORDS = 50000
MAX_CHAR_SEQUENCE_LENGTH = 55
MAX_WORD_SEQUENCE_LENGTH = 25
EMBEDDINGS_INDEX = 'glove.twitter.27B.200d'

# Model
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 256
