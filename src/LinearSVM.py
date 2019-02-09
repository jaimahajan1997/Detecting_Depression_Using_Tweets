from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import nltk
from nltk import *
def baseform(input):
    ans=[]
    for i in word_tokenize(input):
        ans.append(nltk.wordnet.WordNetLemmatizer().lemmatize(i))
        #Alternative ans.append(PorterStemmer().stem(i))
    return ans
#f = open("../datasets/classifier2_datasetA_pickle.pkl", "rb")
f = open("../datasets/classifier1_datasetA_Combined_pickle.pkl", "rb")

filter_tweets = pickle.load(f)
y = pickle.load(f)

print(len(filter_tweets))
print(len(y))

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
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm



model = svm.LinearSVC(C = 0.23, verbose  = 1)
model.fit(train_X, train_y)
file_linear = open("model_linearSVM.pkl", 'wb')
pickle.dump(model, file_linear)
predict_test = model.predict(test_X)
predict_train = model.predict(train_X)
print("precision, recall, fscore - ", precision_recall_fscore_support(test_y, predict_test, average='macro'))
print("accuracy on test - ", accuracy_score(test_y, predict_test))
print("accuracy on train", accuracy_score(train_y, predict_train))
print(confusion_matrix(test_y, predict_test))
'''
Best Output till Now
precision, recall, fscore -  (0.7379726472039786, 0.7369942059943855, 0.7372860191483493, None)
accuracy on test -  0.738327904451683
accuracy on train 0.9612739775606225
[[311 128]
 [113 369]]
'''
'''
precision, recall, fscore -  (0.997375782109526, 0.9931120096474427, 0.995221372446307, None)
accuracy on test -  0.9964483030781374
accuracy on train 0.9976979742173112
[[3809    1]
 [  17 1241]]
'''