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
    return ans
f = open("../datasets/classifier2_datasetA_pickle.pkl", "rb")
filter_tweets = pickle.load(f)
y = pickle.load(f)

print(len(filter_tweets))
print(len(y))

train_X, test_X, train_y, test_y = train_test_split(filter_tweets, y, test_size = 0.25, random_state = 42)
vectorizer = TfidfVectorizer(min_df =  0.001,ngram_range = (1,2),tokenizer=baseform)
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
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier(verbose = 1)
# apply(train_X)
# mode
print("saksham")
model.fit(train_X, train_y)
print("saksham")
file_Rbf = open("model_GradientBoost.pkl", 'wb')
pickle.dump(model, file_Rbf)
predict_test = model.predict(test_X)
predict_train = model.predict(train_X)
print("precision, recall, fscore - ", precision_recall_fscore_support(test_y, predict_test, average='macro'))
print("accuracy on test - ", accuracy_score(test_y, predict_test))
print("accuracy on train", accuracy_score(train_y, predict_train))
print(confusion_matrix(test_y, predict_test))
'''
Best Output till Now
precision, recall, fscore -  (0.7335467349551856, 0.7327432206353557, 0.7329976748451374, None)
accuracy on test -  0.7339847991313789
accuracy on train 0.8635541078537822
[[310 129]
 [116 366]]
'''