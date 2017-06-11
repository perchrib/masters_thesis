import os
from global_constants import TRAIN_DATA_DIR
TXT_DIR = TRAIN_DATA_DIR
save_dirs = [os.path.join(TXT_DIR, "pan14-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan15-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan16-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan17-author-profiling-training-dataset-english-txt/")]


def line_counter(file):
    f = open(file, 'r')
    content = [line.strip() for line in f]
    f.close()
    return len(content) - 1


def gender_counter(file):
    f = open(file, 'r')
    content = f.readline().split(":::")[1]
    return content


if __name__ == "__main__":
    total_tweets = 0
    male = 0
    female = 0
    for pan_set in save_dirs:
        pan_files = os.listdir(pan_set)
        pan_version = 0

        for file in pan_files:
            if ".txt" in file:
                file = pan_set + file
                total_tweets += line_counter(file)
                pan_version += line_counter(file)
                gender = gender_counter(file)


        print(pan_set[14:19], " ", pan_version)

    print("Total Tweets: ", total_tweets)

