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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
clf_NearestNeighbour = KNeighborsClassifier(n_neighbors=3,n_jobs=1)
clf_NearestNeighbour.fit(train_X, train_y)
predict_test = clf_NearestNeighbour.predict(test_X)
predict_train = clf_NearestNeighbour.predict(train_X)
clf_mnb = open("model_NearestNeighbour.pkl", 'wb')
pickle.dump(clf_NearestNeighbour, clf_mnb)
print("precision, recall and fscore - ", precision_recall_fscore_support(test_y, predict_test, average='macro'))
print("accuracy on test - ", clf_NearestNeighbour.score(test_X, test_y))
print("accuracy on train - ", clf_NearestNeighbour.score(train_X, train_y))

print(confusion_matrix(test_y, predict_test))

'''
Best Output till now
precision, recall and fscore -  (0.6884458586059461, 0.6791321279029102, 0.6780126507001135, None)
accuracy on test -  0.6840390879478827
accuracy on train -  0.7875497647484618
Confusion Matrix
[[252 187]
 [104 378]]
'''