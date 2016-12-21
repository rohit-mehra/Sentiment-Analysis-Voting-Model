import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier  # Wrapper to include scikitlearn algo within nltk
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
import pickle
import os


class Train:
    def __init__(self, file1, file2):
        self.classifiers = []

        self.input_data(file1, file2)
        self.set_featureWords()
        self.train()

    def input_data(self, file1, file2):
        try:
            # f1 = input("Enter file path to pos.txt: ")
            # f2 = input("Enter file path to neg.txt: ")
            f1 = file1
            f2 = file2
            pos_rev_str = open(f1, "r", encoding="ISO-8859-1").read()
            neg_rev_str = open(f2, "r", encoding="ISO-8859-1").read()

            all_str = pos_rev_str + neg_rev_str

            pos_list = [(list(word_tokenize(p)), "pos") for p in pos_rev_str.split('\n')]
            neg_list = [(list(word_tokenize(n)), "neg") for n in neg_rev_str.split('\n')]
            if pos_list.__len__() - neg_list.__len__() not in range(-50, 50):
                print("Number of Positive data should be equal to Negative data +50/-50....")
                quit()
                # self.input_data()
            else:
                self.all_list = pos_list[:5500] + neg_list[:5500]
                random.shuffle(self.all_list)

                # #  tagging
                tagged_words = nltk.pos_tag(word_tokenize(all_str))
                # #  Screening
                self.all_words = []
                allowed = ["J", "R"]
                for w_tuple in tagged_words:
                    if w_tuple[1][0] in allowed:
                        self.all_words.append(w_tuple[0].lower())
        except FileNotFoundError:
            print("Enter the Correct Path...")
            # self.input_data()

    def set_featureWords(self):

        all_words = nltk.FreqDist(self.all_words)
        self.feature_words = []
        c = all_words.most_common(2000)
        for i in range(0, 2000):
            if c[i][0] not in ['i', 'u', 'ur', 'you', 'your', 'it', 'they', 'he', 'she', 'her', 'him', 'im']:
                self.feature_words.append(c[i][0])
        self.set_pickle("feature_words", self.feature_words)

    def set_pickle(self, oname, obj):
        i = 0
        if not os.path.exists("pickles"):
            os.makedirs("pickles")
        while True:
            filename = oname + str(i) + ".pickle"
            if not os.path.isfile("pickles/" + filename):
                fw_file = open("pickles/" + filename, "wb")
                pickle.dump(obj, fw_file)
                fw_file.close()
                break
            i += 1

    def get_pickle(self, fname):
        filename = fname + ".pickle"
        if os.path.isfile("pickles/" + filename):
            fw_file = open("pickles/" + filename, "rb")
            data = pickle.load(fw_file)
            fw_file.close()
            return data

    def find_features(self, review_words):
        review_words = [w.lower() for w in review_words]
        review_words = set(review_words)
        features = {}
        for f in self.feature_words:
            features[f] = (f in review_words)
        return features

    def train(self):
        feature_list = [(self.find_features(rev_words), category) for (rev_words, category) in self.all_list]
        training_list = feature_list[:11000]

        NBclassifier = nltk.NaiveBayesClassifier.train(training_list)
        self.set_pickle("NBclassifier", NBclassifier)

        MNBc = SklearnClassifier(MultinomialNB())
        MNBc.train(training_list)
        self.set_pickle("MNBc", MNBc)

        BNBc = SklearnClassifier(BernoulliNB())
        BNBc.train(training_list)
        self.set_pickle("BNBc", BNBc)

        LogisticRegressionC = SklearnClassifier(LogisticRegression())
        LogisticRegressionC.train(training_list)
        self.set_pickle("LogisticRegressionC", LogisticRegressionC)

        LinearSVCc = SklearnClassifier(LinearSVC())
        LinearSVCc.train(training_list)
        self.set_pickle("LinearSVCc", LinearSVCc)
