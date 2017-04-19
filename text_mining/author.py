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