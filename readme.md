## Required python packages

- keras v2.0.2
- tensorflow v1.0
- h5py
- numpy
- nltk (corpus content must downloaded using nltk.download)
- pandas (for logging)
- keras_diagram (for logging)
- langdetect (for language detection preprocessing?)


## Other
- To use GloVe embeddings, download glove.twitter.27B.zip and from https://nlp.stanford.edu/projects/glove/
    * Place this in the <i>embeddings_native</i> directory and run parse_glove() in preprocessors/embedding_parser.py