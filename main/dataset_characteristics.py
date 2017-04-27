from helpers.global_constants import TEXT_DATA_DIR
from text_mining.helpers import get_data, word_tokenize
from text_mining.dataset_characteristics import Characteristics, equal_token_count, unequal_token_count, most_common, lower, stopwords_counter, pos_tag_counter
from text_mining.data_plot import Visualizer


import nltk

MALE_COLOR = "C0"
FEMALE_COLOR = "C1"


def tag_plotter(male_tags, female_tags, tag_type):
    # TODO maybe take visualizer_1 and visualizer_2 in same figure using subplot
    visualizer_1 = Visualizer(title='50 Most Frequent ' + tag_type + ' of Male', xlabel=tag_type, ylabel="Frequency")
    male_most_freq_tags_50 = most_common(male_tags, 50)
    visualizer_1.plot_one_dataset_token_counts(male_most_freq_tags_50, MALE_COLOR, "male")
    visualizer_1.save_plot()

    visualizer_2 = Visualizer(title='50 Most Frequent ' + tag_type + ' of Female', xlabel=tag_type, ylabel="Frequency")
    female_most_freq_tags_50 = most_common(female_tags, 50)
    visualizer_2.plot_one_dataset_token_counts(female_most_freq_tags_50, FEMALE_COLOR, "female")
    visualizer_2.save_plot()

    visualizer_1_and_2 = Visualizer(title='50 Most Frequent ' + tag_type + ' by Gender', xlabel=tag_type, ylabel="Frequency")
    visualizer_1_and_2.plot_one_dataset_token_counts(male_most_freq_tags_50, MALE_COLOR, "male", subplot=True)
    visualizer_1_and_2.plot_one_dataset_token_counts(female_most_freq_tags_50, FEMALE_COLOR, "female", subplot=True)
    visualizer_1_and_2.save_plot()

    visualizer_3 = Visualizer(title='50 Most Common Frequent ' + tag_type + ' by Gender', xlabel=tag_type, ylabel="Frequency")
    male, female = equal_token_count(male_tags, female_tags, 50)
    visualizer_3.plot_two_dataset_token_counts(male, female)
    visualizer_3.save_plot()

    visualizer_4 = Visualizer(title='50 Most Frequent Distinct ' + tag_type + ' by Gender', xlabel=tag_type, ylabel="Frequency")
    male, female = unequal_token_count(male_tags, female_tags, 50)
    visualizer_4.plot_one_dataset_token_counts(male, MALE_COLOR, "male", subplot=True)
    visualizer_4.plot_one_dataset_token_counts(female, FEMALE_COLOR, "female", subplot=True)
    visualizer_4.save_plot()


def plot_two_counters(male_counter, female_counter, counter_type):
    visualizer_1 = Visualizer(title="Frequency of " + counter_type, xlabel=counter_type, ylabel="Frequency")
    male, female = equal_token_count(male_counter, female_counter)
    visualizer_1.plot_two_dataset_token_counts(male, female)
    visualizer_1.save_plot()

def plot_text_length(male_counter, female_counter, length_type):
    visualizer_1 = Visualizer(title='Length of Tweets in ' + length_type, xlabel='Tweet Length', ylabel='Number of Tweets')
    visualizer_1.plot_avg_length_of_texts(male_counter, MALE_COLOR, "male", subplot=True)
    visualizer_1.plot_avg_length_of_texts(female_counter, FEMALE_COLOR, "female", subplot=True)
    visualizer_1.save_plot()


if __name__ == '__main__':
    authors, female_texts, male_texts = get_data(TEXT_DATA_DIR)
    print "Retrieved Data..."
    print "Number of Tweets Total: ", len(female_texts) + len(male_texts)
    print "Number of Male Tweets: ", len(male_texts)
    print "Number of Female Tweets: ", len(female_texts)
    male_data = Characteristics(male_texts)
    female_data = Characteristics(female_texts)
    print "Characteristics Objects Created..."

    # tag_plotter(lower(male_data.hashtag_count), lower(female_data.hashtag_count), tag_type="Hashtags")
    # tag_plotter(lower(male_data.mention_count), lower(female_data.mention_count), tag_type="Mentions")
    #
    plot_two_counters(male_data.emoticon_count, female_data.emoticon_count, counter_type="Emoticons")
    plot_two_counters(male_data.twitter_syntax_token_count, female_data.twitter_syntax_token_count, counter_type="Twitter Syntax Tokens")
    plot_text_length(male_data.length_of_text_char_count, female_data.length_of_text_char_count, "Characters")
    plot_text_length(male_data.length_of_text_word_count, female_data.length_of_text_word_count, "Words")

    #plot_two_counters(male_data.stopwords_count, female_data.stopwords_count, counter_type="Stopwords")
    #plot_two_counters(stopwords_counter(male_texts), stopwords_counter(female_texts), counter_type="Stopwords")

    # import time
    # start = time.time()
    #
    # plot_two_counters(pos_tag_counter(word_tokenize(male_texts)), pos_tag_counter(word_tokenize(female_texts)), counter_type="POS-tags")
    # end = time.time()
    # seconds = end - start
    # m, s = divmod(seconds, 60)
    # print m, "minutes ", s, " seconds"



    #print(pos_tag_counter(word_tokenize(["At eight o'clock on Thursday film morning word line test best beautiful Ram Aaron design", "This is a test"])))