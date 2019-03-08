from django.shortcuts import render
import tweepy
import datetime
from paralleldots import set_api_key, get_api_key
from paralleldots import similarity, taxonomy, sentiment, emotion, abuse
import nltk
import json
import numpy
from django.views.decorators.csrf import csrf_exempt,csrf_protect
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
import os
import pickle

def model_make():
	BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	filepath = os.path.join(BASE_DIR, 'twitter-airline-sentiment/Tweets.csv')

	df = pd.read_csv(filepath)
	n = df.shape[0]
	y = df['airline_sentiment'].values
	data = df['text']

	corpus = []
	for i in range(n):
		review = re.sub('@[a-zA-Z]+', '', data[i])
		review = re.sub('[^a-zA-Z]', ' ', review)
		review = review.lower()
		review = review.split()
		wnl = WordNetLemmatizer()
		review = [wnl.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
		review = ' '.join(review)
		corpus.append(review)

	cv = CountVectorizer(analyzer="word")
	X = cv.fit_transform(corpus).toarray()

	classifier = RandomForestClassifier(n_estimators=200)
	classifier.fit(X, y)
	return classifier
	
#classifier = model_make()

consumer_key = "t36hUioLChtYtHR54qFXVCXKJ"
consumer_secret = "vP2ScbQMf3QV2LtiW5LHszVbXjBzapiuvWXraZfPENoZNdtMSc"
access_key = "994162189480218624-0PLtGzr9End24dhXCWyDPAHp9zAfxTv"
access_secret = "N48xj3bN4t8YZSJQCHHwvQWySEvH2xa75X4I9CG5k1f0d"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    template = 'index.html'
    context = {}
    return render(request,template,context)

def query(datas):
    tmp = []
    tweets_for_csv = [data.full_text for data in datas]
    for j in tweets_for_csv:
        tmp.append(j)
    return tmp

def get_date(datas):
    date = []
    for data in datas:
        t = data.created_at
        date.append(t.strftime('%m/%d/%Y'))
    return date

def get_time(datas):
    time = []
    for data in datas:
        t = data.created_at
        time.append(t.strftime('%H:%M:%S'))
    return time
	
def get_sentiment(tweets,classifier):
    n= len(tweets)
    corpus = []
    for i in range(n):
        review = re.sub('@[a-zA-Z]+', '', tweets[i])
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        wnl = WordNetLemmatizer()
        review = [wnl.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(analyzer="word")
    X = cv.fit_transform(corpus).toarray()

    y_pred = classifier.predict(X)
    return y_pred

def sentiment_analysis(tweets):
    set_api_key("4U0rm3Hboel2L0HqxMPvErNT67FQZvr4gBrxwrY1geg")
    get_api_key()
    sentiment_ana = []
    for text in tweets:
        value = sentiment(text)
        sentiment_value = sentiment(text)
        try:
            values1 = sentiment_value['sentiment']
            sentiment_ana.append(values1)
        except:
            sentiment_ana.append("don't know")
    return sentiment_ana	

def get_entity(tweets):
    nlp = spacy.load('en')
    n = len(tweets)
    entity = []
    for i in range(n):
        review = re.sub('\W',' ',tweets[i])
        doc = nlp(review)
        enty = []
        for j in range(len(doc.ents)):
            enty.append(doc.ents[j])
            enty.append(doc.ents[j].label_)
        entity.append(enty)
    return entity

@csrf_exempt
def get_tweets(request):
	if request.method =="POST":
		post = request.POST
		user = post.get('user')
		datas = api.user_timeline(screen_name=user,tweet_mode="extended")
		name = datas[0].user.name
		discription = datas[0].user.description
		followers = datas[0].user.followers_count
		location = datas[0].user.location
		tweets=query(datas)
		date = get_date(datas)
		time = get_time(datas)
		filepath = os.path.join(BASE_DIR, 'twitter-airline-sentiment/mymodel.sav')
		load_model = pickle.load(open(filepath,'rb'))
		sentiments = get_sentiment(tweets,load_model)
		#sentiments = sentiment_analysis(tweets)
		entities = get_entity(tweets)
		template = 'tweets.html'
		context ={'tweets':zip(tweets,date,time,entities,sentiments),'follower':followers,'location':location,'name':name,'discription':discription}
		return render(request,template,context)
	else:
		return render(request,'index.html',{})

