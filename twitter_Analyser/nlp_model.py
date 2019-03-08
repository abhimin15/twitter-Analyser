import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Tweets.csv')
n=df.shape[0]
y = df['airline_sentiment'].values
data = df['text']

corpus = []
for i in range(n):
    review = re.sub('@[a-zA-Z]+','',data[i])
    review = re.sub('[^a-zA-Z]',' ',review)
    review = review.lower()
    review = review.split()
    wnl = WordNetLemmatizer()
    review = [wnl.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
cv = CountVectorizer(analyzer = "word")
X = cv.fit_transform(corpus).toarray()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state =0)

classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_test,y_test)
con = confusion_matrix(y_test,y_pred)