import xml.etree.cElementTree as ET
import os
from preprocessors.parser import Parser

__author__ = "Per Berg"

XML_DIR = "../data/xml/"
TXT_DIR = "../data/txt/"

__dirs__ = [os.path.join(XML_DIR, "pan14-author-profiling-training-dataset-english-xml/"),
            os.path.join(XML_DIR, "pan15-author-profiling-training-dataset-english-xml/"),
            os.path.join(XML_DIR, "pan16-author-profiling-training-dataset-english-xml/"),
            os.path.join(XML_DIR, "pan17-author-profiling-training-dataset-english-xml/")]

save_dirs = [os.path.join(TXT_DIR, "pan14-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan15-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan16-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan17-author-profiling-training-dataset-english-txt/")]


data_loss = {}
DATA_PARSER = Parser()
TRUTH = {}
characters = set()


def get_all_files(dir):
    files = os.listdir(dir)  # type: List
    truth_file = files.pop(files.index('truth.txt'))
    return files, truth_file


def write_all_files(files, __dir__, save_dir):
    """
    Write tweets to .txt files
    :param files:
    :param __dir__: directory with xml files containing tweets
    :param save_dir: directory to save parsed .txt files to
    :return: list containing not accessed files for removal of .txt files, and number of successfully written files
             and written tweets
    """
    files_not_accessed = []
    total_files_stored = 0
    total_tweets_stored = 0
    for filename in files:
        if '.txt' in filename:
            pass
        if '.xml' in filename:
            author = filename[:-4]

            if not os.path.exists(save_dir + author + ".txt"):

                parsed_data_in_file = get_parsed_data_from_xml_file(filename, __dir__)
                if not parsed_data_in_file:
                    files_not_accessed.append(filename)
                else:
                    # create_file_to_author_and_set_author_data(filename, save_dir)
                    create_file_to_author_and_set_author_data(filename, save_dir)
                    write_data_to_file(parsed_data_in_file, filename, save_dir)
                    total_files_stored += + 1
                    total_tweets_stored += len(parsed_data_in_file)
    return files_not_accessed, total_files_stored, total_tweets_stored


def get_parsed_data_from_xml_file(file, dir):
    current_file = dir + file
    all_parsed_data = []
    try:
        tree = ET.parse(current_file)
        root = tree.getroot()
        documents = root.findall('.//document') if 'pan15' in dir else root.findall('.//documents/document')
        lost_data = 0
        for doc in documents:
            id = doc.get('id')
            url = doc.get('url')
            raw_data = doc.text

            if raw_data:
                clean_text = DATA_PARSER.clean_html(raw_data)
                # chars = [[c for c in word] for word in clean_text.split()]
                # chars = sum(chars, [])
                # for c in chars:
                #    characters.add(c)

                if clean_text not in all_parsed_data: # Avoid duplicate tweets
                    if clean_text:  # Avoid adding empty strings
                        all_parsed_data.append(clean_text)
            else:
                lost_data += 1
                data_loss[file] = lost_data
        return all_parsed_data

    except ET.ParseError:
        print("Parse Error: ")
        # file_not_accessed.append(file)


def create_file_to_author_and_set_author_data(author_file, save_dir):
    author = author_file[:-4]
    path = save_dir + author + ".txt"
    if not os.path.exists(path):
        file = open(path, 'a')
        gender = TRUTH[author][0]
        age = TRUTH[author][1]
        file.write(author + ":::" + gender + ":::" + age)
        print("File: ", path, " created")
        file.close()


def write_data_to_file(data, author_file, save_dir):
    author = author_file[:-4]
    path = save_dir + author + ".txt"
    counter = 0
    if os.path.exists(path):
        file = open(path, 'a')
        for tweet in data:
            try:
                file.write('\n'+tweet)
            except UnicodeEncodeError:
                file.write('\n' + tweet.encode('utf-8'))
        file.close()


def generate_truth(file, __dir__):
    txt_file = open(__dir__ + file, 'r')
    for line in txt_file:
        line_strip = line.strip()
        author_data = line_strip.split(":::")
        TRUTH[author_data[0]] = [author_data[1].upper(), author_data[2].upper()]

    txt_file.close()


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except Exception as e:
            raise e


if "__main__" == __name__:
    total_files_not_accessed = 0
    total_authors = 0
    total_tweets = 0
    for i in range(len(__dirs__)):
        __dir__ = __dirs__[i]
        save_dir = save_dirs[i]
        create_directory(save_dir)

        files, truth = get_all_files(__dir__)
        generate_truth(truth, __dir__)
        files_not_accessed, written_files, tweets_stored = write_all_files(files, __dir__, save_dir)
        total_files_not_accessed += len(files_not_accessed)
        total_authors += written_files
        total_tweets += tweets_stored


        if i == len(__dirs__)-1:
            print("\n\r", "#"*30)
            print("Number of Tweets", total_tweets)
            print("Number of Authors: ", total_authors)
            print("\n\r", "#"*30)
            print("Total Files Not Accessed: ", total_files_not_accessed)
            print("Tweets Not Available: ", sum(data_loss.values()))