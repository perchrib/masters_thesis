from helpers.global_constants import TEST_DATA_DIR, REM_STOPWORDS, LEMMATIZE, REM_EMOTICONS, REM_PUNCTUATION

# Prediction type
GENDER = 'gender'
AGE = 'age'
C_BAG_OF_WORDS = 'bow'
C_TF_IDF = 'tfidf'
C_TF_IDF_DISSIMILARITY = 'tfidf_dissimilarity'
C_BAG_OF_WORDS_DISSIMILARITY = 'bow_dissimilarity'
SAVE_FEATUREMODEL = False

PREDICTION_TYPE = GENDER


# Log directory
LOGS_DIR = '../logs/document_level_classification'
MODEL_DIR = '../models/document_level_classification'
AUTOENCODER_DIR = '../models/document_level_classification/autoencoders'
TEST_DATA_DIR = TEST_DATA_DIR

# Autoencoder
DIM_REDUCTION = False
DIM_REDUCTION_SIZE = 400

if not DIM_REDUCTION:
    DIM_REDUCTION_SIZE = None


#############################################
# Model (TUNING PARAMETERS HERE!)

# set TYPE = "" when not using "base"

MODEL_TYPE = "final"
LAYERS = [[2048, 1024, 512]]
# Can be represent as one structure ie [128,64] or multiple structure [[100, 50, 20], [22, 44, 22],]


# from random_search import generate_random_layers
#LAYERS = [[2048], [4096], [2048, 1024], [4096, 1024]]
# Regularization
DROPOUT = 0
L1 = 0
L2 = 0
LAYER_PENALTY = 0

# Text pre-processing
FEATURE_MODEL = C_BAG_OF_WORDS
SENTIMENT_FEATURE = False
EMOTICON_FEATURE = False

EXPERIMENTS = True
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
    REM_STOPWORDS:True,
    LEMMATIZE: False,
    REM_PUNCTUATION: False,
    REM_EMOTICONS: False
}


def check_if_zero(integer):
    if integer == 0:
        return None
    return integer


def get_constants_info(n_gram=None, vocabulary_size=None):
    # parameters

    dropout = check_if_zero(DROPOUT)
    l1 = check_if_zero(L1)
    l2 = check_if_zero(L2)

    info = ["\n--- Regularisation ---\n",
                  "\t-Dropout: %s" % dropout,
                  "\t-L1: %s" % l1,
                  "\t-L2: %s" % l2,
                  "\n--- Feature Info ---\n",
                  "\t-Vocabulary Size: %s" % vocabulary_size,
                  "\t-Embedding: %s" % FEATURE_MODEL,
                  "\t-Ngram: %s" % (n_gram, ),
                  "\t-Autoencoder: %s" % DIM_REDUCTION,
                  "\t-Reduction size: %s" % DIM_REDUCTION_SIZE]
    return info
