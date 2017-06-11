from helpers.helper_functions import load_pickle
import os
pickle_path = "../models/word_embedding_classification/word_index"
pickle_file = "23.05.2017_18:12:28_BiLSTM_punct_em.pkl"
glove_path = "../embeddings_index/glove.twitter.27B.200d.pkl"
import numpy as np

def main_plot(final_embeddings, reverse_dictionary):
    def plot_with_labels(low_dim_embs, labels, filename='tsne_twitter'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(30, 30))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        #print('RUNNING')
        plt.savefig(filename)
        print("Data Was Plotted")
        #print('Running_1')

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        #print("Labels: ", labels)
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


if __name__ == "__main__":
    print("Load Word Index")
    word_index = load_pickle(os.path.join(pickle_path, pickle_file))
    reversed_word_index = dict(zip(word_index.values(), word_index.keys()))
    print("Load GLoVe Word Embeddings")
    word_embeddings = load_pickle(glove_path)
    values = sorted(word_index.values())
    final_embeddings = []
    reversed_dictionary = {}
    print("Create Dictionaries for Plotting")

    j = 0
    for value in values:
        word = reversed_word_index[value]
        try:
            final_embeddings.append(word_embeddings[word])
            if word == 'm':
                word = "MENTION"
            elif word == 'p':
                word = 'IMAGE'
            elif word == 'u':
                word = "URL"
            elif word == 'h':
                word = "HASHTAG"
            reversed_dictionary[j] = word
            j += 1
        except KeyError:
            print(word, " not in glove...")
    final_embeddings = np.asarray(final_embeddings)
    final_embeddings = final_embeddings.reshape(j, 200)
    print("Plot")
    main_plot(final_embeddings, reversed_dictionary)