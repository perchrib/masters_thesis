import xml.etree.cElementTree as ET
import os
from nltk.tokenize import TweetTokenizer
import re
__author__ = "Per Berg"

XML_DIR = "../data/xml/"
TXT_DIR = "../data/txt/"

__dirs__ = [os.path.join(XML_DIR, "pan16-author-profiling-training-dataset-english-xml/"),
            os.path.join(XML_DIR, "pan17-author-profiling-training-dataset-english-xml/"),
            os.path.join(XML_DIR, "pan14-author-profiling-training-dataset-english-xml/")]

save_dirs = [os.path.join(TXT_DIR, "pan16-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan17-author-profiling-training-dataset-english-txt/"),
             os.path.join(TXT_DIR, "pan14-author-profiling-training-dataset-english-txt/")]


class Parser:
    def __init__(self):
        self.tknzr = TweetTokenizer()

    def parse_content(self, content):
        content = self.clean_HTML(content)
        content = content.lower()
        content = self.replace('url', ' ~ ', content)
        content = self.replace('pic', ' P ', content)
        content = self.replace('@', ' A ', content)
        content = self.replace('#', ' H ', content)
        content = self.do_join(content)
        return content

    def clean_HTML(self, content):
        """
        Strips the input string from html code
        :param content: string containing html syntax
        :return: a string with no html syntax
        """
        # strip html tags
        p = re.compile(r'<.*?>')
        content = p.sub('', content)
        # clean characters which are not defined in regex
        content = re.sub(r'[^\x00-\x7f]', r'', content)
        content = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content)[0]
        # split all words in tokens + convert some asci into strings. ie can#39;t = can't
        tokens = self.tknzr.tokenize(content)
        content = ' '.join(tokens)
        return content


    def replace(self, remove, replace, content):
        """
        :param remove: specifies what will be removed: 'url'=urls, '@'=mentions, '#'=hashtag, 'pic'=picture urls
        :param replace: replace a string or character
        :param content: the string that will be modified
        :return: the modified string :param content:
        """
        if remove == "url":
            content = re.sub(r'(?:(http://|https://)|(www\.)|(http|httphttp|https) :/ / )(\S+\b/?)([!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]*)(\s|$)', replace, content)
        elif remove == "@":
            content = re.sub(r'@([a-z0-9_]+)', replace, content)
        elif remove == "#":
            content = re.sub(r'#([a-z0-9_]+)', replace, content)
        elif remove == 'pic':
            content = re.sub(r'(pic .twitter.com/|pic.twitter.com/)(\w+)', replace, content)
        return content

    def do_join(self, content):
        """
        :param content: text string
        :returns: a string with no more than one whitespace between words
        """
        for _ in range(10):
            content = " ".join(content.split())
        return content

data_loss = {}
data_accessed = []
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
    :return: list containing not accessed files for removal of .txt files
    """
    files_not_accessed = []

    # Create directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in files:
        # print("\nFilename: ", filename)
        if '.txt' in filename:
            pass
        if '.xml' in filename:
            author_id = filename[:-4]
            create_file_to_author_and_set_author_data(filename, save_dir)
            parsed_data_in_file = get_parsed_data_from_xml_file(filename, __dir__)
            if not parsed_data_in_file:
                files_not_accessed.append(filename)
            else:
                write_data_to_file(parsed_data_in_file, filename, save_dir)

        # print("AUTHOR: ", author_id)

    return files_not_accessed

def get_parsed_data_from_xml_file(file, dir):
    current_file = dir + file
    all_parsed_data = []
    # print("File Path => ", current_file, "\n")
    try:
        tree = ET.parse(current_file)
        root = tree.getroot()
        documents = root.findall('.//documents/document')
        lost_data = 0
        for doc in documents:
            id = doc.get('id')
            url = doc.get('url')
            raw_data = doc.text
            if raw_data:
                clean_text = DATA_PARSER.parse_content(raw_data)
                chars = [[c for c in word] for word in clean_text.split()]
                chars = sum(chars, [])
                for c in chars:
                    characters.add(c)
                if clean_text not in all_parsed_data: # Avoid duplicate tweets
                    all_parsed_data.append(clean_text)
                data_accessed.append(id)
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
    if os.path.exists(path):
        file = open(path, 'a')
        for tweet in data:
            file.write('\n'+tweet)
        file.close()


def generate_truth(file, __dir__):
    txt_file = open(__dir__ + file, 'r')
    for line in txt_file:
        line_strip = line.strip()
        author_data = line_strip.split(":::")
        TRUTH[author_data[0]] = author_data[1:]

    txt_file.close()


if "__main__" == __name__:
    for i in range(len(__dirs__)):
        __dir__ = __dirs__[i]
        save_dir = save_dirs[i]

        files, truth = get_all_files(__dir__)
        generate_truth(truth, __dir__)
        files_not_accessed = write_all_files(files, __dir__, save_dir)

        print("Total Files Not Accessed: ", len(files_not_accessed))
        print("Tweets Available", len(data_accessed))

        print("Vocabulary: ", characters, " Length: ", len(characters))
        total = 0

        for file in data_loss:
            total += data_loss[file]

        print("Tweets Not Available: ", total)

        for file in files_not_accessed:
            print("EMPTY FILE: ", file[:-4] + ".txt")  # -4 to remove '.xml'
            txt_file = save_dir + file[:-4] + ".txt"
            print(txt_file)
            os.remove(txt_file)
            print("REMOVED FILE")

