

# From char dataset formatting
def format_dataset_char_level_sentences(texts, labels, metadata):
    """
    Split into training set and validation set.
    Also splits each tweet into sentences to encode the sentences by themselves.
    :param texts: list of tweets
    :param labels: list of tweet labels
    :param metadata: list of dictionaries containing age and gender for each tweet
    :return:
    """

    print('\n-------Creating character set...')
    all_text = ''.join(texts)

    chars = set(all_text)
    print('Total Chars: %i' % len(chars))
    char_index = dict((char, i) for i, char in enumerate(chars))

    data = np.ones((len(texts), MAX_SENTENCE_LENGTH, MAX_CHAR_SENT_LENGTH), dtype=np.int64) * -1

    labels = to_categorical(np.asarray(labels))  # convert to one-hot label vectors

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    for i, tweet in enumerate(texts):
        sentences = sent_tokenize(tweet)
        for j, sent in enumerate(sentences):
            if j < MAX_SENTENCE_LENGTH:
                for k, char in enumerate(sent[:MAX_CHAR_SENT_LENGTH]):
                    # data[i, j] = char_index[char]
                    data[i, j, MAX_CHAR_SENT_LENGTH-1-k] = char_index[char]  # Input reversed

    # shuffle and split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    metadata = [metadata[i] for i in indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    meta_train = metadata[:-nb_validation_samples]

    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    meta_val = metadata[-nb_validation_samples:]

    return x_train, y_train, meta_train, x_val, y_val, meta_val, char_index


# From dataset preperation for splitting into test set as well
def split_dataset(data, labels, metadata, data_type_is_string=False):
    """
    Given correctly formatted dataset, split into training, validation and test
    :param data: formatted dataset, i.e., sequences of char/word indices
    :return: training set, validation set, test set and metadata
    """
    np.random.seed(SEED)
    # shuffle and split the data into a training set and a validation set
    if data_type_is_string:
        data, labels, indices = shuffle(data, labels)
    else:
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

    metadata = [metadata[i] for i in indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    nb_test_samples = int(TEST_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples-nb_test_samples]
    y_train = labels[:-nb_validation_samples-nb_test_samples]
    meta_train = metadata[:-nb_validation_samples-nb_test_samples]

    x_val = data[-nb_validation_samples-nb_test_samples:-nb_test_samples]
    y_val = labels[-nb_validation_samples-nb_test_samples:-nb_test_samples]
    meta_val = metadata[-nb_validation_samples-nb_test_samples:-nb_test_samples]

    x_test = data[-nb_test_samples:]
    y_test = labels[-nb_test_samples:]
    meta_test = data[-nb_test_samples:]

    return x_train, y_train, meta_train, x_val, y_val, meta_val, x_test, y_test, meta_test