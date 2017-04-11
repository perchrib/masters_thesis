from nltk.tokenize import TweetTokenizer
import re


class Parser():
    def __init__(self):
        self.tknzr = TweetTokenizer()

    def parse_all_content(self, content):
        content = self.clean_HTML(content)
        content = content.lower()
        content = self.replace('url', ' ~ ', content)
        content = self.replace('pic', ' P ', content)
        content = self.replace('@', ' A ', content)
        content = self.replace('#', ' H ', content)
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
        content = self.do_join(content)
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
        content = self.do_join(content)
        return content

    def do_join(self, content):
        """
        :param content: text string
        :returns: a string with no more than one whitespace between words (max 10 spaces)
        """
        for _ in range(10):
            content = " ".join(content.split()).strip()
        return content

    def generate_character_vocabulary(self, texts):
        pass

    def generate_word_vocabulary(self, texts):
        pass

