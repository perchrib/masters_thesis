from __future__ import print_function
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk import WordNetLemmatizer, pos_tag
import re

import string
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

    def lowercase(self, texts):
        print("Lowercasing texts")
        return [t.lower() for t in texts]  # Lower_case

    def replace_all_twitter_syntax_tokens(self, texts):
        # Raise error if texts not lists
        if type(texts) is not list:
            raise Exception("Parser must be passed a list of texts")

        modified_texts = self.replace(texts, url=URL_REPLACE, pic=PIC_REPLACE, mention=MENTION_REPLACE, hashtag=HASHTAG_REPLACE)

        print("Replacing Internet terms - Done")
        return modified_texts

    def remove_all_twitter_syntax_tokens(self, texts):
        # Raise error if texts not lists
        if type(texts) is not list:
            raise Exception("Parser must be passed a list of texts")

        modified_texts = self.replace(texts, url="", pic="", mention="", hashtag="")

        print("Removing Internet terms - Done")
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

        return modified_texts


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

        print("Removing stopwords - Done")
        return parsed_texts

    def lemmatize(self, texts):
        """
        Given list of texts. Lemmatize all text terms
        :param texts: list of texts
        :return: lemmatized texts
        """
        lemmatized_texts = []
        count = 0
        for t in texts:  # type: str
            count += 1
            terms = word_tokenize(t)
            pos_tags = pos_tag(terms)  # POS-tags needed to determine correct root form
            lemmatized_terms = []
            for i in range(len(terms)):
                try:
                    l_term = self.lemmatizer.lemmatize(word=pos_tags[i][0], pos=get_wordnet_pos(pos_tags[i][1])).encode('utf-8')
                    lemmatized_terms.append(l_term)
                except Exception:
                    # UnicodeDecodeError
                    continue

            # [self.lemmatizer.lemmatize(word=pos_tags[i][0], pos=get_wordnet_pos(pos_tags[i][1])).encode('utf-8') for i in range(len(terms))]
            lemmatized_texts.append(" ".join(lemmatized_terms))

        print("Lemmmatization - Done")
        return lemmatized_texts

    def remove_punctuation(self, texts):
        new_texts = []
        for t in texts:
            EMOTICON_ID = "EMO1231298736"
            #print("Before: ", t)
            emoticons_in_t = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|(?:<3)', t)
            t = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)|(?:<3)', EMOTICON_ID, t)
            t = ' '.join(word.strip(string.punctuation) for word in t.split())
            new_t = []
            #print("AFTER: ", t)
            for word in t.split():
                if word == EMOTICON_ID:
                    word = emoticons_in_t[0]
                    del emoticons_in_t[0]
                new_t.append(word)
            new_texts.append(' '.join(new_t))

        print("Removing punctuations - Done")
        return new_texts

    def remove_emoticons(self, texts):
        new_texts = []
        for t in texts:
            t = re.sub('(?::|;|=)(?:-)?(?:\)|\(|D|P)|(?:<3)', "", t)
            new_texts.append(t)

        print("Removing emoticons - Done")
        return new_texts

    def remove_texts_shorter_than_threshold(self, texts, labels, metadata, threshold=2):
        """
        Remove texts shorter than threshold (length in characters) from list of texts, labels and metadata
        :param modified_texts:
        :param modified_labels:
        :param modified_metadata:
        :param threshold:
        :return:
        """

        removal_count = 0

        modified_texts = []
        modified_labels = []
        modified_metadata = []

        for i in range(len(texts)):
            if len(texts[i]) >= threshold:
                modified_texts.append(texts[i])
                modified_labels.append(labels[i])
                modified_metadata.append(metadata[i])
            else:
                removal_count += 1

        print("Removed %i empty tweets" % removal_count)

        return modified_texts, modified_labels, modified_metadata, removal_count

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

