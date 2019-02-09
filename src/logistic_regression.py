from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import nltk
from nltk import *

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
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
clf_logistic = LogisticRegression(random_state = 42, verbose = 1, solver = 'saga', C = 1, max_iter = 100)
scaler = StandardScaler()
scaler.fit(train_X, train_y)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)
clf_logistic.fit(train_X, train_y)
file_logistic = open("model_logisticRegression.pkl", 'wb')
pickle.dump(clf_logistic, file_logistic)
predict_test = clf_logistic.predict(test_X)
predict_train = clf_logistic.predict(train_X)
print("precision, recall, fscore - ", precision_recall_fscore_support(test_y, predict_test, average='macro'))
print("accuracy on test - ", accuracy_score(test_y, predict_test))
print("accuracy on train", accuracy_score(train_y, predict_train))
print(confusion_matrix(test_y, predict_test))
'''
Best Output till now
precision, recall, fscore -  (0.7451143195033014, 0.7350849251883288, 0.7351393322405314, None)
accuracy on test -  0.739413680781759
accuracy on train 0.997828447339848
Confusion Matrix
[[282 157]
 [ 83 399]]
'''