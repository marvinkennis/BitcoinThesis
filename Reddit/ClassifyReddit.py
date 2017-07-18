from __future__ import division
import statsmodels as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import time 
from dateutil import parser
import datetime
from scipy.stats.stats import pearsonr 
import re
import string
from collections import defaultdict
from bs4 import BeautifulSoup
import sys
from sklearn import svm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import math
import unicodecsv
from sklearn.metrics import confusion_matrix
import logging
import spacy
from datetime import date, timedelta as td


from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import csv
import json
from collections import Counter
from sklearn import metrics

from os.path import join, dirname
from nltk import ngrams
import nltk

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
np.set_printoptions(threshold=np.nan)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")


    return plt

def summary_to_sentences(summary, tokenizer, totalsents, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(summary.strip())
    #
    # 2. Loop over each sentence
    sentences = []

    for raw_sentence in totalsents:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( summary_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def createCorpus(df1):
	rc = df1.shape[0]
	#rc2 = df2.shape[0]
	corpus = []
	corpus_ns = []
	sentiments = []
	d2v_sents = []
	sentences = []
	ratings =[]
	for i in range(0,rc):
		cratings = []
		splitwords = unicode(df1.iloc[i]['summ_items_wordstring'].split(' '))
		#print splitwords
		for j in range(0,5):
			cratings.append(df1.iloc[i][4+j])
		cratings.append(df1.iloc[i][10])

		ratings.append(cratings)
		if float(df1.iloc[i]['mturk_avg']) > 0:
			print 1 
			sentiments.append(1)
			corpus.append(df1.iloc[i]['summ_items_wordstring'])
			corpus_ns.append(df1.iloc[i]['summ_items_ns_wordstring'])

			d2v_sents.append('D2V Sent')

		elif float(df1.iloc[i]['mturk_avg'])< 0:
			print 0
			sentiments.append(0)
			corpus.append(df1.iloc[i]['summ_items_wordstring'])
			corpus_ns.append(df1.iloc[i]['summ_items_ns_wordstring'])

			d2v_sents.append('d2v sent')

		else:
			pass

			

	return [corpus, corpus_ns, sentiments, d2v_sents,ratings]



def evalModel(model,target,x):
	mismatched = []

	scores = cross_val_score(model, x, target, cv=10)
	predicted = cross_val_predict(model, x, target, cv=10)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	print confusion_matrix(target, predicted)
	print 'accuracy'
	print(metrics.accuracy_score(target, predicted))
	print 'precision'
	print(metrics.average_precision_score(target, predicted))
	print 'recall'
	print(metrics.recall_score(target, predicted))

	curve = metrics.precision_recall_curve(target, predicted, pos_label=None, sample_weight=None)
	print 'AUC'
	print metrics.roc_auc_score(target, predicted, average='macro', sample_weight=None)
	positive = 0
	negative = 0
	for j in range(0,len(target)):

		if target[j] == 1:
			positive +=1
		else:
			negative += 1

		if target[j] != predicted[j]:
			mismatched.append(j)
	print '----------'
	print positive
	print negative
	return mismatched




logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)


dataset = 'TrainingDataReddit.csv'
df1 = pd.read_csv(dataset)
'''
dataset2 = 'TrainingDataReddit.csv'
df2 = pd.read_csv(dataset2)
'''


rc = df1.shape[0]

corpora = createCorpus(df1)



print len(corpora[0])
transformer = TfidfTransformer()

#tfidfvect = TfidfVectorizer(min_df=1, ngram_range=(1,1))
unigram_vectorizer = CountVectorizer(min_df=1)
bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b', min_df=1)
trigram_vectorizer = CountVectorizer(ngram_range=(1,3),token_pattern=r'\b\w+\b', min_df=1)

#vectorizer = TfidfVectorizer(smooth_idf=False)
v = unigram_vectorizer.fit_transform(corpora[0]).toarray()

x = unigram_vectorizer.fit_transform(corpora[0]).toarray()
x2 = bigram_vectorizer.fit_transform(corpora[0]).toarray()
x3 = trigram_vectorizer.fit_transform(corpora[0]).toarray()

tf_idf_x = transformer.fit_transform(x).toarray()
tf_idf_x2 = transformer.fit_transform(x2).toarray()
tf_idf_x3 = transformer.fit_transform(x3).toarray()



target = corpora[2]

model = LogisticRegression()
model.fit(v, corpora[2])
mmarray = evalModel(model,target,x)


#-----------------
#-----------------

#-----------------
#-----------------

print len(mmarray)
zeros = []
for c in mmarray:
	#print corpora[0][c]
	if corpora[-1][c][-1] == 0:
		zeros.append(corpora[-1][c][-1])

	#print '----------------------------'
	#print '----------------------------'
print len(zeros)




		


nlp = spacy.load('en')
output = defaultdict(lambda: defaultdict(int))


dataset = 'AllReddit.csv'
df1 = pd.read_csv(dataset)

rc = df1.shape[0]
print rc
for i in range(0,rc):


	summary = df1.iloc[i]['content']
	ts = df1.iloc[i]['timestamp']

	key = datetime.datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d')
	publisher = df1.iloc[i]['subreddit'].lower().replace(".","")
	print key
	publisher_total = str(publisher)+"_total"
	if '2015' in key:
		print 'yes 2015'
		output[key][publisher_total] += 1
		output[key]['total'] += 1

		print 'nlping summary'
		summ = nlp(unicode(summary))
		summ_items = []
		wordstring = ""
		for token in summ:
			summ_items.append([token.lemma_, token.pos_])

		summ_items = [w for w in summ_items if w[1] != unicode('PUNCT')]
			

		for item in summ_items:
			wordstring += str(item[0].encode('ascii', 'ignore'))+" "

		#print wordstring
		print 'creating test data'
		test_data = unigram_vectorizer.transform([wordstring]).toarray()
		classif = model.predict(test_data)

		publisher_cumulative_sent = str(publisher)+"_csent"
		
		publisher_positive = str(publisher)+"_positive"
		publisher_negative = str(publisher)+"_negative"

		output[key][publisher_cumulative_sent] += classif[0]
		if output[key]['csent']:
			pass
		else:
			output[key]['csent'] = 0 

		if output[key]['negative']:
			pass
		else:
			output[key]['negative'] = 0 
			
		if output[key]['positive']:
			pass
		else:
			output[key]['positive'] = 0 

		if classif[0] > 0:
			output[key][publisher_positive] += 1
			output[key]['positive'] += 1
			output[key]['csent'] += 1
		else:
			output[key][publisher_negative] += 1
			output[key]['negative'] += -1
			output[key]['csent'] -=  1
	else:
		pass


print len(output)
print output['2015-02-02']


d1 = date(2015, 1, 1)
d2 = date(2015, 12, 31)

delta = d2 - d1

datelist =[]
for i in range(delta.days + 1):
    datelist.append(str(d1 + td(days=i))) 


marketdata = 'MarketData/USD-BTCavg.csv'
df = pd.read_csv(marketdata)

rc = df.shape[0]

marketresults = {}
for i in range(0,rc):
	key = df.iloc[i]['Date']
	avg = df.iloc[i]['Average']
	lag = 1
	try:
		avg_lag = df.iloc[i]['Average']
		vol_lag = df.iloc[i]['TotalVolume']
		change = df.iloc[i]['Average'] - df.iloc[i+1]['Average']
		perc_change = (df.iloc[i]['Average'] - df.iloc[i+1]['Average'])/df.iloc[i+1]['Average']*100
		perc_change_vol = (df.iloc[i]['TotalVolume'] - df.iloc[i+1]['TotalVolume'])/df.iloc[i+1]['TotalVolume']*100
		volumechange = df.iloc[i]['TotalVolume'] - df.iloc[i+1]['TotalVolume']
	except:
		change = 0
		perc_change = 0
		perc_change_vol = 0
	volume = df.iloc[i]['TotalVolume']


	marketresults[key] = [avg, change, perc_change, volume, volumechange, avg_lag, vol_lag, perc_change_vol]




csents = []

avgs = []
avg_changes = []

vol_changes = []
vols = []

perc_changes_avg = []
perc_changes_vol = []
total_csent = 0
tcentarray = []
for d in list(marketresults.keys()):
	total_csent += int(output[d]['csent'])
	tcentarray.append(output[d]['csent'])
a = np.array(tcentarray)
p = np.percentile(a, 10) 
avg_csent = float(total_csent) / float(len(list(marketresults.keys())))
prev_sentiment = 0
for d in list(marketresults.keys()):

	#if output[d]['csent'] < p:
	csents.append(output[d]['csent'])
	avgs.append(marketresults[d][5])
	avg_changes.append(marketresults[d][1])
	vol_changes.append(marketresults[d][4])
	vols.append(marketresults[d][6])
	perc_changes_avg.append(marketresults[d][2])
	perc_changes_vol.append(marketresults[d][7])

	prev_sentiment = output[d]['csent']
	row = [d,output[d]['csent'],output[d]['negative'],output[d]['positive'],marketresults[d][0],marketresults[d][1],marketresults[d][2],marketresults[d][3],marketresults[d][4],marketresults[d][5],marketresults[d][6],marketresults[d][7]]
	with open('111111FINAL_REDDIT_GRANGER.csv', 'a') as testfile:
		csv_writer = csv.writer(testfile)
		csv_writer.writerow(row)

		'''
		csents.append(prev_sentiment)
		changes.append(marketresults[d][5])
		'''














