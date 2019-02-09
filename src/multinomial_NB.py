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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
clf_multinomialnb = MultinomialNB(alpha=0.059)
clf_multinomialnb.fit(train_X, train_y)
predict_test = clf_multinomialnb.predict(test_X)
predict_train = clf_multinomialnb.predict(train_X)
clf_mnb = open("model_multinomialNB.pkl", 'wb')
pickle.dump(clf_multinomialnb, clf_mnb)
print("precision, recall and fscore - ", precision_recall_fscore_support(test_y, predict_test, average='macro'))
print("accuracy on test - ", clf_multinomialnb.score(test_X, test_y))
print("accuracy on train - ", clf_multinomialnb.score(train_X, train_y))

print(confusion_matrix(test_y, predict_test))
'''
Best Output till now
precision, recall and fscore -  (0.7570850202429149, 0.7430788570780442, 0.7428983972662078, None)
accuracy on test -  0.748099891422367
accuracy on train -  0.9952949692363373
Confusion Matrix
[[279 160]
 [ 72 410]]
'''
'''
precision, recall and fscore -  (0.970861601992252, 0.959380802757366, 0.9649471772099736, None)
accuracy on test -  0.9741515390686661
accuracy on train -  0.9781636411470666
[[3767   43]
 [  88 1170]]

'''