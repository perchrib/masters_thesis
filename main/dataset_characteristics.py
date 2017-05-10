from __future__ import print_function, division
from helpers.global_constants import TEXT_DATA_DIR
from text_mining.helpers import get_data, word_tokenize, seperate_authors_by_gender
from text_mining.dataset_characteristics import Characteristics, equal_token_count, unequal_token_count, most_common, \
    lower, stopwords_counter, pos_tag_counter, least_common
from text_mining.data_plot import Visualizer
import numpy as np
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

    visualizer_3 = Visualizer(title='50 Most Frequent ' + tag_type + ' by Gender', xlabel=tag_type, ylabel="Frequency")
    visualizer_3.plot_one_dataset_token_counts(male_most_freq_tags_50, MALE_COLOR, "male", subplot=True)
    visualizer_3.plot_one_dataset_token_counts(female_most_freq_tags_50, FEMALE_COLOR, "female", subplot=True)
    visualizer_3.save_plot()

    visualizer_4 = Visualizer(title='50 Most Common Frequent ' + tag_type + ' by Gender', xlabel=tag_type, ylabel="Frequency")
    male, female = equal_token_count(male_tags, female_tags, 50)
    visualizer_4.plot_two_dataset_token_counts(male, female)
    visualizer_4.save_plot()

    visualizer_5 = Visualizer(title='50 Most Frequent Distinct ' + tag_type + ' by Gender', xlabel=tag_type, ylabel="Frequency")
    male, female = unequal_token_count(male_tags, female_tags, 50)
    visualizer_5.plot_one_dataset_token_counts(male, MALE_COLOR, "male", subplot=True)
    visualizer_5.plot_one_dataset_token_counts(female, FEMALE_COLOR, "female", subplot=True)
    visualizer_5.save_plot()


def plot_two_counters(male_counter, female_counter, counter_type):
    visualizer = Visualizer(title="Frequency of " + counter_type, xlabel=counter_type, ylabel="Frequency")
    male, female = equal_token_count(male_counter, female_counter)
    visualizer.plot_two_dataset_token_counts(male, female)
    visualizer.save_plot()


def plot_text_length(male_counter, female_counter, length_type):
    visualizer = Visualizer(title='Length of Tweets in ' + length_type, xlabel='Tweet Length', ylabel='Number of Tweets')
    visualizer.plot_avg_length_of_texts(male_counter, MALE_COLOR, "male", subplot=True)
    visualizer.plot_avg_length_of_texts(female_counter, FEMALE_COLOR, "female", subplot=True)
    visualizer.save_plot()


def plot_pos_tags(male_counter, female_counter, pos_tag_type):
    visualizer = Visualizer(title='Frequency of ' + pos_tag_type, xlabel='Pos-Tags',
                              ylabel='Number of Tweets')
    male, female = equal_token_count(male_counter, female_counter)
    visualizer.plot_two_dataset_token_counts(male, female)
    visualizer.save_plot()


def plot_frequent_tokens(male_counter, female_counter, counter_type):
    visualizer = Visualizer(title=counter_type +' Common Tokens', xlabel='Tokens', ylabel='Frequency')
    visualizer.plot_one_dataset_token_counts(male_counter, MALE_COLOR, "male", subplot=True)
    visualizer.plot_one_dataset_token_counts(female_counter, FEMALE_COLOR, "female", subplot=True)
    visualizer.save_plot()


def plot_dataset_distribution_men_female(num_male, num_female, ylabel):
    visualizer = Visualizer(title="Distribution of " + ylabel +  " by Gender", xlabel="Gender", ylabel="Number of " + ylabel)
    visualizer.plot_simple_histograms(["Male", "Female"], [num_male, num_female])
    visualizer.save_plot(filename=ylabel, topic="gender-distribution")


def plot_length_of_tweet_by_gender_and_the_total(male, female, total):
    s = np.random.uniform(0, 100, 1000)
    s[999] = 300
    visualizer = Visualizer(title="Number of Tweets by Gender", xlabel="",
                            ylabel="Number of Tweets")
    visualizer.plot_boxplot(distributions=[male, female, s], xlabel=["Male", "Female", "Both"])
    visualizer.save_plot(filename="Number of Tweets by Gender")


def find_avg_and_median_tweet_amount_by_author(authors):
    avg_total = 0
    med_total = []
    med = 0
    avg = 0
    for author in authors:
        avg_total += len(author.tweets)
        med_total.append(len(author.tweets))
    med_total = sorted(med_total)
    if len(authors) % 2 == 0:

        i = int(len(authors)/2)
        med = round((med_total[i] + med_total[i-1])/2, 3)
    else:
        i = len(authors) // 2
        med = med_total[i]
    avg = round(avg_total / len(authors), 3)
    return avg, med

def get_distribution_of_tweets(authors):
    total = []
    for author in authors:
        total.append(len(author.tweets))
    return np.asarray(total)

if __name__ == '__main__':
    authors, female_texts, male_texts = get_data(TEXT_DATA_DIR)
    female_authors, male_authors = seperate_authors_by_gender(authors)
    print("Retrieved Data...")
    print("*"*20, "--Twitter--", "*"*20)
    print("Number of Tweets Total: ", len(female_texts) + len(male_texts))
    print("Number of Male Tweets: ", len(male_texts))
    print("Number of Female Tweets: ", len(female_texts))
    print("\n")
    print("*"*20, "--Authors--", "*"*20)
    print("Number of Different Authors: ", len(authors))
    print("Number of Male Authors: ", len(male_authors))
    print("Number of Female Authors: ", len(female_authors))
    print("\n")
    avg, med = find_avg_and_median_tweet_amount_by_author(authors)
    print("AVG of the Amount of Tweets by Author: ", avg)
    print("Median of the Amount of Tweets by Author: ", med)
    #male_data = Characteristics(male_texts)
    #female_data = Characteristics(female_texts)

    print("Characteristics Objects Created...")

    #plot_dataset_distribution_men_female(len(male_authors), len(female_authors), ylabel="Authors")
    #plot_dataset_distribution_men_female(len(male_texts), len(female_texts), ylabel="Tweets")

    #print(get_distribution_of_tweets(male_authors))

    # plot_length_of_tweet_by_gender_and_the_total(male_data.length_of_text_char_count.values(),
    #                                              sorted(get_distribution_of_tweets(female_authors)),
    #                                              get_distribution_of_tweets(authors))




    # tag_plotter(lower(male_data.hashtag_count), lower(female_data.hashtag_count), tag_type="Hashtags")
    # tag_plotter(lower(male_data.mention_count), lower(female_data.mention_count), tag_type="Mentions")
    # #
    #
    # plot_two_counters(male_data.emoticon_count, female_data.emoticon_count, counter_type="Emoticons")
    # plot_two_counters(male_data.twitter_syntax_token_count, female_data.twitter_syntax_token_count, counter_type="Twitter Syntax Tokens")
    # plot_text_length(male_data.length_of_text_char_count, female_data.length_of_text_char_count, "Characters")
    # plot_text_length(male_data.length_of_text_word_count, female_data.length_of_text_word_count, "Words")

    # Frequency of word accurance
    #plot_two_counters(male_data.stopwords_count, female_data.stopwords_count, counter_type="Stopwords")
    #plot_two_counters(stopwords_counter(male_texts), stopwords_counter(female_texts), counter_type="Stopwords")


    #plot_frequent_tokens(male_data.most_common(50), female_data.most_common(50), counter_type="Most")
    #plot_frequent_tokens(male_data.least_common(50), female_data.least_common(50), counter_type="Least")



    import time
    start = time.time()

    # plot simple pos tags
    #plot_two_counters(pos_tag_counter(word_tokenize(male_texts)), pos_tag_counter(word_tokenize(female_texts)), counter_type="POS-tags")

    # plot all pos tags
    #plot_pos_tags(pos_tag_counter(word_tokenize(male_texts), simple_pos_tags=False), pos_tag_counter(word_tokenize(female_texts), simple_pos_tags=False), pos_tag_type="All Pos-Tags Types")
    end = time.time()
    seconds = end - start
    m, s = divmod(seconds, 60)
    print(m, "minutes ", s, " seconds")
