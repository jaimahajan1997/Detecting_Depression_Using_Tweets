#Author - Harsh Pathak
import csv
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import pickle
import sys
from ekphrasis.classes.spellcorrect import SpellCorrector
from ekphrasis.classes.segmenter import Segmenter
import string
dataset_name = 'classifier1_datasetA_Combined' #has 5068 positive(1's)  and 15204 negative samples(0's) = 20272 tweets
#dataset_name = 'classifier2_datasetA' #has 1879 depression, 1805 non depression tweets
tweets = [] #collection of strings
y = [] #sentiment

def read_dataset(filename):
    with open("../datasets/" + filename + ".csv",encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            if(row[0].isdigit() == True):
                sentiment = int(row[0])
                tweet = row[1]
                tweets.append(tweet)
                y.append(sentiment)
    return tweets, y

def preprocess_dataset(tweets, y): 
    """uses ekphrasis API to preprocess the tweets"""

    text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    spell_correction = False,
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
    )
    ynew = []
    filter_tweets = []
    for t in range(0, len(tweets)):
        tokens = text_processor.pre_process_doc(tweets[t])
        newtokens = []
        i = 0
        while(i < len(tokens)):
            try:
                if(tokens[i] == "pic" and tokens[i + 1] == "." and tokens[i + 2] == "twitter"):
                    break
                elif(tokens[i] in ["<url>", "<email>", "<user>", "<money>", "<percent>", "<phone>", "<time>", "<date>", "<number>"]):
                    i += 1
                    continue
                elif(tokens[i] == "<" and tokens[i + 1] == "emoji"):
                    while(tokens[i] != ">"):
                        i += 1
                    i += 1
                else:
                    newtokens.append(tokens[i])
                    i += 1
            except:
                break
        if(len(newtokens) != 0):
            filter_tweets.append(" ".join(newtokens))
            ynew.append(y[t])
    return filter_tweets, ynew		 #tokenizing and other preprocessing #removes emojis

tweets, y = read_dataset(dataset_name)
filter_tweets, y = preprocess_dataset(tweets, y)

f = open("../datasets/" + dataset_name + "_pickle.pkl", "wb")
pickle.dump(filter_tweets, f)
pickle.dump(y, f)
f.close()

print(len(filter_tweets))
print(len(y))



###TEST PREPROCESSING HERE

#sp = SpellCorrector(corpus="twitter")
#seg = Segmenter(corpus="twitter")
#tokenized_tweets2 = []
#print(tokenized_tweets)
#for i in tokenized_tweets:
#	modified_tweet = []
#	for j in i.split(" "):
#		if(j in string.punctuation):
#			modified_tweet.append(j)
#			continue
#		correct_word = sp.correct(j)
#		print(j, " ", correct_word)
#		modified_tweet.append(correct_word)
#	tokenized_tweets2.append(" ".join(modified_tweet))
#tokenized_tweets = tokenized_tweets2
#print((tokenized_tweets))


