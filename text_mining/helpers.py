import os
from author import Author


def flatten(list):
    return [item for sublist in list for item in sublist]

def word_tokenize(texts):
    return flatten(map(lambda x: x.split(), texts))


def get_data(path):
    """
    :param path, the main path containing the dataset
    :return all_authors, a list of the object Author (all authors in the dataset), 
    :return female_texts, a list of all texts, (each tweet is one element) from female gender
    :return male_texts, a list of all texts, (each tweet is one element) from male gender
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

    return all_authors, flatten(female_texts), flatten(male_texts)