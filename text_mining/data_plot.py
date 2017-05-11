import matplotlib.pyplot as plt 
import os
import numpy as np
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

            plt.title(self.title, fontsize=18)
            plt.xlabel(self.xlabel, size=15)
            plt.ylabel(self.ylabel, size=15)
            self.subplot_counter = 211


        plt.grid(True, color="silver")


    def determine_fontsize(self, n_tokens):
        if n_tokens <= 25:
            return 10
        elif n_tokens <= 50:
            return 6
        elif n_tokens <= 100:
            return 5
        return 3


    def plot_two_dataset_token_counts(self, tc1, tc2, tc1_label="male", tc2_label="female"):
        """
        Plot two distributions with two Counter object
        :param tc1: A Counter object for dataset nr 1 (male)
        :param tc2: A Counter object for dataset nr 2 (female)
        :param tc1_label: labelname for dataset nr 1
        :param tc2_label: labelname for dataset nr 1
        
        """
        rotation = 90
        if "emoticons" in self.title.lower() or "emoticons" in self.xlabel.lower():
            rotation = 270
        common_tokens = sorted(tc1.keys())
        font_size = self.determine_fontsize(len(common_tokens))
        tc1_freq = [tc1[token] for token in common_tokens]
        tc2_freq = [tc2[token] for token in common_tokens]

        plt.plot(tc1_freq, label=tc1_label)
        plt.plot(tc2_freq, label=tc2_label, alpha=0.7)

        plt.xticks(range(len(common_tokens)), common_tokens, rotation=rotation, fontsize=font_size)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()


    def plot_one_dataset_token_counts(self, tc, color, label="gender", subplot=False):
        """
        :param tc: Counter object for a dataset
        :param color: color option
        :param label: labelname for the dataset
        :param subplot: True or False, if True, wait for another function call and take another tc dataset and plot
                this in same figure.  
        :return: 
        """
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
        """
        :param counts: Counter object
        :param color: color option 
        :param label: labelname
        :param subplot: True or False, if True, wait for another function call and take another tc dataset and plot
                this in same figure. 
        :return: 
        """
        x_values = counts.keys()
        y_values = counts.values()

        total_elements = sum(counts.values())
        total_sum_of_lengths = reduce(lambda x, y: x+y, map(lambda x: x[0] * x[1], zip(counts.values(), counts.keys())))
        avg_length = round(float(total_sum_of_lengths)/float(total_elements), 2)
        if subplot:
            self.subplot_checker()


        plt.axvline(x=avg_length, color='black', label="avg length " + str(avg_length))
        #print "AVG length: ", avg_length, "total_elements: ", total_elements, " total_sum: ", total_sum_of_lengths

        plt.bar(x_values, y_values, color=color, label=label)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()

    def plot_simple_histograms(self, x_labels, y_labels, color=["C0", "C1"]):
        """
        :param x_labels: list of labels name that will be plottet in the x axis 
        :param y_labels: list of labels values that will be the histogram value in the plot
        """
        if len(x_labels) != len(y_labels):
            raise TypeError("Not Same Length")
        x = np.arange(len(x_labels))

        plt.bar(x, height=y_labels, color=color)
        plt.xticks(x, x_labels)
        plt.grid(False)
        plt.tight_layout()

    def plot_boxplot(self, distribution=None, distributions=None, xlabel=None):
        colors = ["C0", "C1", "C2"]
        #ax = self.figure.add_subplot(111)
        #plt.set_xticklabels(xlabel)
        if distributions:
            bp = plt.boxplot(distributions, patch_artist=True)
        else:
            bp = plt.boxplot(distribution)
        i = 0
        for box in bp['boxes']:
            box.set(linewidth=1)
            box.set_facecolor(colors[i])
            i += 1
        for median in bp['medians']:
            median.set(color='C3', linewidth=2)

        for flier in bp['fliers']:
            flier.set(marker='x', color='C3', alpha=0.5)

        plt.xticks(range(1, len(xlabel) + 1), xlabel)










    def subplot_checker(self):
        plt.subplot(self.subplot_counter)
        if self.subplot_counter == 211:
            plt.title(self.title, fontsize=18)
            plt.ylabel(self.ylabel, size=15)
        else:
            plt.ylabel(self.ylabel, size=15)
            plt.xlabel(self.xlabel, size=15)
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
            plt.savefig(os.path.join(save_dir, filename.lower() + "." + file_format), format=file_format, dpi=quality)

    def show(self):
        plt.show()

    def create_dir(self, save_dir):
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                raise e







