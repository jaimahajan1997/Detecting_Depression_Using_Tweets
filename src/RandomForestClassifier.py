from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import nltk
from nltk.stem import *


#f = open("../datasets/classifier2_datasetA_pickle.pkl", "rb")
f = open("../datasets/classifier1_datasetA_Combined_pickle.pkl", "rb")
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
vectorizer = TfidfVectorizer(ngram_range = (1,2),tokenizer=baseform,max_features=2000)#Remove max features parameter for 2nd classifier
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
clf_RandomForest = RandomForestClassifier(n_estimators=120)
clf_RandomForest.fit(train_X, train_y)
predict_test = clf_RandomForest.predict(test_X)
predict_train = clf_RandomForest.predict(train_X)
clf_mnb = open("model_RandomForest.pkl", 'wb')
pickle.dump(clf_RandomForest, clf_mnb)
print("precision, recall and fscore - ", precision_recall_fscore_support(test_y, predict_test, average='macro'))
print("accuracy on test - ", clf_RandomForest.score(test_X, test_y))
print("accuracy on train - ", clf_RandomForest.score(train_X, train_y))

print(confusion_matrix(test_y, predict_test))

'''
Best Output till now
precision, recall and fscore -  (0.7427837039593208, 0.7260961823835763, 0.7250928997407871, None)
accuracy on test -  0.7318132464712269
accuracy on train -  0.9985522982265653
[[265 174]
 [ 73 409]]
'''
'''
precision, recall and fscore -  (0.9982904364863978, 0.9958942036061073, 0.9970852095192577, None)
accuracy on test -  0.9978295185477506
accuracy on train -  1.0
[[3809    1]
 [  10 1248]]
'''