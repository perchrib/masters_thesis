from nltk.tokenize import TweetTokenizer
from nltk import WordNetLemmatizer
import re


class Parser:
    def __init__(self):
        self.tknzr = TweetTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def replace_all(self, texts):
        # content = self.clean_HTML(content)  # TODO: Kan fjernes?
        modified_texts = [t.lower() for t in texts]  # Lower_case

        modified_texts = self.replace('url', 'U', modified_texts)
        modified_texts = self.replace('pic', 'P', modified_texts)
        modified_texts = self.replace('@', 'M', modified_texts)
        modified_texts = self.replace('#', 'H', modified_texts)

        return modified_texts

    def clean_html(self, content):
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
        content = self.do_join(content)
        return content

    def replace(self, remove, replace, texts):

        """
        :param remove: specifies what will be removed: 'url'=urls, '@'=mentions, '#'=hashtag, 'pic'=picture urls
        :param replace: the string or character which will be placed instead
        :param texts: list of texts to be modified
        :return: the modified string :param content:
        """
        modified_texts = []
        replace = ' ' + replace + ' '
        for txt in texts:
            content = txt
            if remove == "url":
                content = re.sub(r'(?:(http://|https://)|(www\.)|(http|httphttp|https) :/ / )(\S+\b/?)([!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]*)(\s|$)', replace, content)
            elif remove == "@":
                content = re.sub(r'@([a-z0-9_]+)', replace, content)
            elif remove == "#":
                content = re.sub(r'#([a-z0-9_]+)', replace, content)
            elif remove == 'pic':
                content = re.sub(r'(pic .twitter.com/|pic.twitter.com/)(\w+)', replace, content)
            content = self.do_join(content)

            modified_texts.append(content)

        return modified_texts

    def replace_urls(self, texts):
        return self.replace('url', 'U', texts)

    def do_join(self, content):
        """
        :param content: text string
        :returns: a string with no more than one whitespace between words (max 10 spaces)
        """
        for _ in range(10):
            content = " ".join(content.split()).strip()
        return content

    def lemmatize(self, texts):
        """
        Given list of texts. Lemmatize all text terms
        :param texts: list of texts
        :return: lemmatized texts
        """
        lemmatized_texts = []
        for t in texts:  # type: str
            terms = t.split()
            lemmatized_terms = [self.lemmatizer.lemmatize(w) for w in terms]
            lemmatized_texts.append(" ".join(lemmatized_terms))

        return lemmatized_texts

    def generate_character_vocabulary(self, texts):
        pass

    def generate_word_vocabulary(self, texts):
        pass



