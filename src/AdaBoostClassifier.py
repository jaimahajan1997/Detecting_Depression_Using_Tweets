from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import nltk
from nltk.stem import *


f = open("../datasets/classifier2_datasetA_pickle.pkl", "rb")
filter_tweets = pickle.load(f)
y = pickle.load(f)

print(len(filter_tweets))
print(len(y))
def baseform(input):
    ans=[]
    for i in word_tokenize(input):
        ans.append(nltk.wordnet.WordNetLemmatizer().lemmatize(i))
        #Alternative ans.append(PorterStemmer().stem(i))
    return ans
train_X, test_X, train_y, test_y = train_test_split(filter_tweets, y, test_size = 0.25, random_state = 42)
vectorizer = TfidfVectorizer(ngram_range = (1,2),tokenizer=baseform)
vectorizer.fit(train_X, train_y)
print(len(train_X))
print(len(test_X))
train_X = vectorizer.transform(train_X)
test_X = vectorizer.transform(test_X)
print(train_X.shape)
print(test_X.shape)
train_X = train_X.toarray()
test_X = test_X.toarray()

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
clf_ADABoost = AdaBoostClassifier(n_estimators=120, algorithm='SAMME.R')
clf_ADABoost.fit(train_X, train_y)
predict_test = clf_ADABoost.predict(test_X)
predict_train = clf_ADABoost.predict(train_X)
clf_mnb = open("model_ADA_Boost.pkl", 'wb')
pickle.dump(clf_ADABoost, clf_mnb)
print("precision, recall and fscore - ", precision_recall_fscore_support(test_y, predict_test, average='macro'))
print("accuracy on test - ", clf_ADABoost.score(test_X, test_y))
print("accuracy on train - ", clf_ADABoost.score(train_X, train_y))

print(confusion_matrix(test_y, predict_test))

'''
Best Output till now
precision, recall and fscore -  (0.7087518037518037, 0.7051035454021304, 0.7053571110014331, None)
accuracy on test -  0.7079261672095548
accuracy on train -  0.8269996380745567
Confusion Matrix
[[283 156]
 [113 369]]
'''