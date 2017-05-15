import os
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

# Remove: Not used anymore.
# For use with sentence encoder
MAX_SENTENCE_LENGTH = 2  # Max number of sentences to consider when
MAX_CHAR_SENT_LENGTH = 52


# Model
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 128
