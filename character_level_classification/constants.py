
# Prediction type
GENDER = 'gender'
AGE = 'age'
PREDICTION_TYPE = GENDER


# Log directory
LOGS_DIR = '../logs/character_level_classification'
MODEL_DIR = '..models/character_level_classification'

# Text pre-processing
MAX_SEQUENCE_LENGTH = 80

# For use with sentence encoder
MAX_SENTENCE_LENGTH = 2  # Max number of sentences to consider when
MAX_CHAR_SENT_LENGTH = 52



# Model
MODEL_OPTIMIZER = 'adam'
MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRICS = ['accuracy']
NB_EPOCHS = 50
BATCH_SIZE = 256
