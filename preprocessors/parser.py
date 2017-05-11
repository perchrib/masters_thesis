from __future__ import print_function
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk import WordNetLemmatizer, pos_tag
import re
from nltk.corpus import stopwords, wordnet


URL_KEY = 'url'
PIC_KEY = 'pic'
MENTION_KEY = 'mention'
HASHTAG_KEY = 'hashtag'

URL_REPLACE = 'U'
PIC_REPLACE = 'P'
MENTION_REPLACE = 'M'
HASHTAG_REPLACE = 'H'


class Parser:
    def __init__(self):
        self.tknzr = TweetTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def replace_all(self, texts):
        # Raise error if texts not lists
        if type(texts) is not list:
            raise Exception("Parser must be passed a list of texts")

        modified_texts = [t.lower() for t in texts]  # Lower_case
        modified_texts = self.replace(modified_texts, url=URL_REPLACE, pic=PIC_REPLACE, mention=MENTION_REPLACE, hashtag=HASHTAG_REPLACE)

        return modified_texts

    def clean_html(self, content):
        """
        Strips the input string from html code
        :param content: string containing html syntax
        :return: a string with no html  syntax
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
        content = self.do_join(content)
        return content

    def replace(self, texts, **kwargs):

        """
        :param remove: specifies what will be removed: URL_KEY=urls, MENTION_KEY=mentions, HASHTAG_KEY=hashtag, PIC_KEY=picture urls
        :param replace: the string or character which will be placed instead
        :param texts: list of texts to be modified
        :return: the modified string :param content:
        """
        # Raise error if texts not lists
        if type(texts) is not list:
            raise Exception("Parser must be passed a list of texts")

        modified_texts = []
        for txt in texts:
            content = txt
            for remove, replace in kwargs.items():
                replace = ' ' + replace + ' '
                if remove == URL_KEY:
                    content = re.sub(r'(?:(http://|https://)|(www\.)|(http|httphttp|https) :/ / )(\S+\b/?)([!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]*)(\s|$)', replace, content)
                elif remove == MENTION_KEY:
                    content = re.sub(r'@([a-z0-9_]+)', replace, content)
                elif remove == HASHTAG_KEY:
                    content = re.sub(r'#([a-z0-9_]+)', replace, content)
                elif remove == PIC_KEY:
                    content = re.sub(r'(pic .twitter.com/|pic.twitter.com/)(\w+)', replace, content)
                content = self.do_join(content)

            modified_texts.append(content)

        modified_texts = [t.lower() for t in modified_texts]  # Lower_case
        return modified_texts

    def replace_urls(self, texts):
        return self.replace(texts, url='U')

    def do_join(self, content):
        """
        :param content: text string
        :returns: a string with no more than one whitespace between words (max 10 spaces)
        """
        for _ in range(20):
            content = " ".join(content.split()).strip()
        return content


    def remove_stopwords(self, texts):
        """
        
        :param content: list of text strings  
        :return: list of text strings with removed stopwords
        """

        # Raise error if texts not lists
        if type(texts) is not list:
            raise Exception("Parser must be passed a list of texts")

        parsed_texts = []
        stop_words = set(stopwords.words('english'))
        for text in texts:  # type: str
            words = re.split("'| ", text)  # split words with space and apostrophes as delimiters
            sustain_words = [word for word in words if word not in stop_words]
            new_text = " ".join(sustain_words)
            parsed_texts.append(new_text)

        return parsed_texts

    def lemmatize(self, texts):
        """
        Given list of texts. Lemmatize all text terms
        :param texts: list of texts
        :return: lemmatized texts
        """
        lemmatized_texts = []
        for t in texts:  # type: str
            terms = word_tokenize(t)
            pos_tags = pos_tag(terms)  # POS-tags needed to determine correct root form
            lemmatized_terms = [self.lemmatizer.lemmatize(word=pos_tags[i][0], pos=get_wordnet_pos(pos_tags[i][1])) for i in range(len(terms))]
            lemmatized_texts.append(" ".join(lemmatized_terms))

        return lemmatized_texts


    def generate_character_vocabulary(self, texts):
        pass

    def generate_word_vocabulary(self, texts):
        pass

def get_wordnet_pos(treebank_tag):
    """
    Convert Penn Trebank Tag to WordNet Tag
    :param treebank_tag: Tag in format of "NOUN", "VERB"
    :return: Tag in format 's', 'v'; using Wordnet constants
    """

    # ADJ, ADJ_SAT, ADV, NOUN, VERB available as pos constants in wordnet
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('RB'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Return Noun as default for all other pos
