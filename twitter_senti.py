from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import classify as s
import json

# Get this Data from twitter API
ckey = ''
csecret = ''
atoken = ''
asecret = ''


class Listener(StreamListener):
    def on_data(self, data):

        try:
            all_data = json.loads(data)
            tweet = str(all_data["text"])

            #  Calculating Sentiment
            sentiment, conf = s.sentiment(tweet)
            print('Tweet:      ', tweet)
            print('Sentiment:  ', sentiment)
            print('Confidence: ', round(conf * 100, 4), '%')
            print(
                "**********************************************************************")

            #  For graph
            if conf >= .6:
                output = open("twitter-out.txt", "a")
                output.write(sentiment)
                output.write("\n")
                output.close()
        except KeyError:
            pass
        except UnicodeEncodeError:
            pass

        return True

    def on_error(self, status):
        print(status)


auth1 = OAuthHandler(ckey, csecret)

auth1.set_access_token(atoken, asecret)
ourStream = Stream(auth1, Listener())
words = input("Input a Word or a phrase to track :")

ourStream.filter(track=[words])
