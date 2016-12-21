from nltk.classify import ClassifierI
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import re
from nltk.corpus import stopwords


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []

        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)  # list of pos and neg
        data = Counter(votes)
        return data.most_common(1)[0][0]

    def classify_many(self, featuresets):

        return [self.classify(fs) for fs in featuresets]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        data = Counter(votes)
        choice_votes = votes.count(data.most_common(1)[0][0])
        conf = choice_votes / len(votes)

        return round(conf, 3)


class LoadClassifiers:
    def __init__(self):
        self.feature_words = []
        self.classifiers = []
        self.load()

    def get_pickle(self, fname):
        fw_file = open("pickles/" + fname, "rb")
        data = pickle.load(fw_file)
        fw_file.close()
        return data

    def set_pickle(self, oname, obj):
        i = 0
        while True:
            filename = oname + str(i) + ".pickle"
            if not os.path.isfile("pickles/" + filename):
                fw_file = open("pickles/" + filename, "wb")
                pickle.dump(obj, fw_file)
                fw_file.close()
                break
            i += 1

    def load(self):

        feature_words = []
        i = 0
        fname = "feature_words"
        while True:
            filename = fname + str(i) + ".pickle"
            i += 1
            if os.path.isfile("pickles/" + filename):
                feature_words += self.get_pickle(filename)
            else:
                self.feature_words = list(set(feature_words))
                # cachedStopWords = stopwords.words("english")
                # self.feature_words = [word for word in self.feature_words if word not in cachedStopWords]
                break

        names = ["NBclassifier", "MNBc", "BNBc", "LogisticRegressionC", "LinearSVCc"]

        for n in names:
            i = 0
            while True:
                filename = n + str(i) + ".pickle"
                i += 1

                if os.path.isfile("pickles/" + filename):
                    self.classifiers.append(self.get_pickle(filename))
                else:
                    break


l = LoadClassifiers()
vc = VoteClassifier(*l.classifiers)


def find_features(review_words):
    review_words = set(review_words)
    features = {}  # Dictionary of words with value = TRUE/FALSE
    # TRUE if we a match between the input_text and ourFeatureWords
    for f in l.feature_words:
        features[f] = (f in review_words)
    return features


def pre_processing(tweet):
    url = re.findall("(https?://[^\s]+)", tweet)
    for i in url:
        tweet = tweet.replace(i, "")

    url = re.findall("(@.+?(?:\s+|$))", tweet)
    for i in url:
        tweet = tweet.replace(i, "")

    url = re.findall("(&.*?;)", tweet)
    for i in url:
        tweet = tweet.replace(i, "")

    url = re.findall("(!+?!)", tweet)
    for i in url:
        tweet = tweet.replace(i, "")
    url = re.findall("(\.+?\.)", tweet)
    for i in url:
        tweet = tweet.replace(i, " ")

    tweet = tweet.replace("RT", "", 1)
    tweet = re.sub(r"[^\w\.\?']", ' ', tweet)
    tweet = tweet.strip()
    tweet = ' '.join(tweet.split())

    return tweet


def sentiment(tweet):
    vs = SentimentIntensityAnalyzer()
    senti = vs.polarity_scores(tweet)
    input_text = pre_processing(tweet)

    input_word_list = [w.lower() for w in word_tokenize(input_text)]
    feats = find_features(
        input_word_list)  # would return matching feature words input_words vs. words used for training

    # sending a dictionary and getting a category returned from the classify function ({features},category)
    emotion = vc.classify(feats)
    conf = vc.confidence(feats)

    # Voting and normalizing
    votes = {'pos': 0, 'neg': 0, 'neu': 0}

    if emotion == 'pos' and senti['pos'] != 0:
        votes['pos'] += conf
        votes['pos'] += senti['pos']

    elif emotion == 'neg' and senti['neg'] != 0:
        votes['neg'] += conf
        votes['neg'] += senti['neg']

    else:
        if senti['pos'] != 0 and senti['neu'] - senti['pos'] < .2:
            votes['pos'] = 1.0
        if senti['neg'] != 0 and senti['neu'] - senti['neg'] < .2:
            votes['neg'] = 1.0

    votes['neu'] += senti['neu']

    emotion = max(votes, key=votes.get)
    conf = votes[emotion]
    if conf > 1:
        conf = 1.0

    return emotion, conf
