import os

class Characteristics:
    """
    :param male_authors, list containing texts with gender male
    :param female_authors, , list containing texts with gender female
    :param authors, list containing all the data as the object Authors
    """
    def __init__(self, male_authors, female_authors, authors):
        self.male = male_authors
        self.female = female_authors
        self.authors = authors


class Author:
    def __init__(self, id, gender, pan_version):
        self.id = id
        self.gender = gender
        self.tweets = []
        self.pan_version = pan_version

    def add_text(self, txt):
        self.tweets = txt

    def number_of_tweets(self):
        return len(self.tweets)

    def tweet_average_length(self, type='word'):
        """
        calculate the average tweet length with regard to amount of tokens or characters 
        :param type, can be either 'token' or 'char'
        :return average length of a tweet 
        """
        counter = 0
        for tweet in self.tweets:
            if type == 'token':
                counter += len(tweet.split())
            elif type == 'char':
                counter += len(tweet)

        return round(float(counter)/float(len(self.tweets)), 2)

def get_data(path):
    """
    :param path, the main path containing the dataset
    :return all_authors, a list of the object Author (all authors in the dataset), 
    :return female_texts, a list of all texts from female gender
    :return male_texts, a list of all texts from male gender
    """
    directories = map(lambda x: path + x + "/", filter(lambda x: 'pan' in x, os.listdir(path)))
    all_authors = []
    female_texts = []
    male_texts = []

    for dir in directories:
        files = map(lambda x: dir + x, filter(lambda x: ".txt" in x, os.listdir(dir)))
        for file in files:
            f = open(file, "r")
            content = [line.strip() for line in f]
            author_data = content.pop(0).split(":::")
            author_id = author_data[0]
            author_gender = author_data[1]
            new_author = Author(author_id, author_gender, dir[12:17])
            new_author.add_text(content)
            all_authors.append(new_author)
            if author_gender == 'MALE':
                male_texts.append(new_author.tweets)
            elif author_gender == 'FEMALE':
                female_texts.append(new_author.tweets)

    return all_authors, sum(female_texts, []), sum(male_texts, [])



def display_dataset_statistics(texts):
    """
    Given a dataset as a list of texts, display statistics: Number of tweets, avg length of characters and tokens.
    :param texts: List of string texts
    """

    # Number of tokens per tweet
    tokens_all_texts = list(map(lambda tweet: tweet.split(" "), texts))
    avg_token_len = reduce(lambda total_len, tweet_tokens: total_len + len(tweet_tokens), tokens_all_texts,
                           0) / len(
        tokens_all_texts)

    # Number of characters per tweet
    char_length_all_texts = list(map(lambda tweet: len(tweet), texts))
    avg_char_len = reduce(lambda total_len, tweet_len: total_len + tweet_len, char_length_all_texts) / len(
        texts)

    print("Number of tweets: %i" % len(texts))
    print("Average number of tokens per tweet: %f" % avg_token_len)
    print("Average number of characters per tweet: %f" % avg_char_len)
