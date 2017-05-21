# Prediction types
GENDER = 'gender'
AGE = 'age'

# Class names
MALE = 'MALE'
FEMALE = 'FEMALE'

# Identifiers
TRAIN = "train"
TEST = "test"

TRAIN_ACC = "Training Accuracy"
TRAIN_LOSS = "Training Loss"
VAL_ACC = "Validation Accuracy"
VAL_LOSS = "Validation Loss"

CHAR = "char"
DOC = "doc"
WORD = "word"


# Directories
TRAIN_DATA_DIR = '../data/train/'
TEST_DATA_DIR = '../data/test/'

CROWDFLOWER_CSV_PATH = '../data/csv/gender-classifier-DFE-791531.csv'

EMBEDDINGS_NATIVE_DIR = '../embeddings_native/'
EMBEDDINGS_INDEX_DIR = '../embeddings_index/'

# Filter constants
REM_STOPWORDS = 'REM_STOPWORDS'
LEMMATIZE = 'LEMMATIZE'
REM_PUNCTUATION = 'REM_PUNCTUATION'
REM_EMOTICONS = 'REM_EMOTICONS'
LOWERCASE = 'LOWERCASE'

REM_INTERNET_TERMS = 'REM_INTERNET_TERMS'


# Dataset split fractions
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Used in prec, rec, f1 score dict
OVERALL_MACRO = 'Overall Macro'  # Used in prec, rec, f1 score dict for macro average
OVERALL_MICRO = 'Overall Micro'  # Used in prec, rec, f1 score dict for micro average
