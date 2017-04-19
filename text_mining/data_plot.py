import matplotlib.pyplot as plt 


class Visualizer():
    def __init__(self, **kwargs):
        self.title = kwargs['title']
        self.xlabel = kwargs['xlabel']
        self.ylabel = kwargs['ylabel']
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

    def plot_two_dataset_token_counts(self, t1, t2, t1_label=None, t2_label=None):
        common_tokens = t1.keys()
        t1_freq = [t1[token] for token in common_tokens]
        t2_freq = [t2[token] for token in common_tokens]

        ylabel = "Counts"
        plt.grid(True, color="silver")

        plt.plot(t1_freq, label=t1_label)
        plt.plot(t2_freq, label=t2_label)
        plt.legend(loc='upper left', frameon=False)
        plt.xticks(range(len(common_tokens)), common_tokens, rotation=270)

        plt.show()
