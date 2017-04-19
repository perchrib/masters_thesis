from config.global_constants import TEXT_DATA_DIR
from text_mining.helpers import get_data
from text_mining.dataset_characteristics import Characteristics, equal_token_count
from text_mining.data_plot import Visualizer

import nltk

if __name__ == '__main__':
    authors, female_texts, male_texts = get_data(TEXT_DATA_DIR)
    print "Retrieved Data..."
    print "Number of Tweets Total: ", len(female_texts) + len(male_texts)
    print "Number of Male Tweets: ", len(male_texts)
    print "Number of Female Tweets: ", len(female_texts)
    male_data = Characteristics(male_texts)
    female_data = Characteristics(female_texts)
    print "Characteristics Objects Created..."
    # conditional
    male_tokens = male_data.most_common_tokens(50)
    female_tokens = female_data.most_common_tokens(50)

    male_emoticons = male_data.emoticon_count
    female_emoticons = female_data.emoticon_count

    male, female = equal_token_count(male_emoticons, female_emoticons)

    print "\nMale\n", len(male), male
    print "\nFemale\n", len(male), female

    visualizer = Visualizer(title="Emoticon Distribution", xlabel="Emoticons", ylabel="Frequency")
    visualizer.plot_two_dataset_token_counts(male, female, t1_label="male", t2_label="female")



    #common_tokens = [w[0] for w in male_tokens if w[0] in map(lambda x: x[0], female_tokens)]
    #rint len(common_tokens), " : ",common_tokens
    #print male_tokens[0]
    #print female_tokens



    #male_data.plot(50, cumulative=None, title="Cumulative Distribution of 50 Most Frequent Words by Male ")
    #female_data.plot(50, cumulative=None, title="Cumulative Distribution of 50 Most Frequent Words by Female ")

    #print "Most Common Male\n", male_data.token_counter()
    #print "\n Most Common Female\n", female_data.token_counter()




