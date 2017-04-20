import matplotlib.pyplot as plt 
import os
save_dir = "../text_mining/data_visualization_figs/"
file_format = 'png'#'eps'
quality = 750 #1200 high
class Visualizer():
    def __init__(self, **kwargs):
        if kwargs:

            self.title = kwargs['title']
            self.xlabel = kwargs['xlabel']
            self.ylabel = kwargs['ylabel']

            self.figure = plt.figure()

            plt.title(self.title)
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)

            self.subplot_counter = 211



        plt.grid(True, color="silver")


    def determine_fontsize(self, n_tokens):
        if n_tokens <= 25:
            return 10
        elif n_tokens <= 50:
            return 6
        elif n_tokens <= 100:
            return 3


    def plot_two_dataset_token_counts(self, tc1, tc2, tc1_label="male", tc2_label="female"):
        rotation = 90
        if "emoticons" in self.title.lower() or "emoticons" in self.xlabel.lower():
            rotation = 270
        common_tokens = sorted(tc1.keys())
        font_size = self.determine_fontsize(len(common_tokens))
        tc1_freq = [tc1[token] for token in common_tokens]
        tc2_freq = [tc2[token] for token in common_tokens]

        plt.plot(tc1_freq, label=tc1_label)
        plt.plot(tc2_freq, label=tc2_label)

        plt.xticks(range(len(common_tokens)), common_tokens, rotation=rotation, fontsize=font_size)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()


    def plot_one_dataset_token_counts(self, tc, color, label="gender", subplot=False):
        tokens = sorted(tc.keys())
        if subplot:
            self.subplot_checker()
        font_size = self.determine_fontsize(len(tokens))
        tc_freq = [tc[token] for token in tokens]
        plt.plot(tc_freq, color=color, label=label)
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=font_size)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()

    def plot_avg_length_of_texts(self, counts, color, label="gender", subplot=False):
        x_values = counts.keys()
        y_values = counts.values()

        if subplot:
            self.subplot_checker()

        #plt.axvline(x=avg_length)

        plt.bar(x_values, y_values, color=color, label=label)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()


    def subplot_checker(self):
        plt.subplot(self.subplot_counter)
        if self.subplot_counter == 211:
            plt.title(self.title)
        self.subplot_counter += 1


    def save_plot(self, filename=None, topic=None):
        self.create_dir(save_dir)
        if not topic and not filename:
            sub_dir = os.path.join(save_dir, self.xlabel + "/").lower()
            self.create_dir(sub_dir)
            plt.savefig(os.path.join(sub_dir, self.title.lower() + "." + file_format), format=file_format, dpi=quality)
        elif topic and filename:
            sub_dir = os.path.join(save_dir, topic + "/").lower()
            self.create_dir(sub_dir)
            plt.savefig(os.path.join(sub_dir, filename.lower() + "." + file_format), format=file_format, dpi=quality)
        elif filename:
            plt.savefig(os.path.join(save_dir, filename.lower() + "." +  file_format), format=file_format, dpi=quality)

    def show(self):
        plt.show()

    def create_dir(self, save_dir):
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                raise e







