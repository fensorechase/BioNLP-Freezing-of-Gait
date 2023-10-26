'''
BMI 550 BioNLP
Week 8;
In-class
@author Abeed Sarker

We will be using the dataset here: https://www.kaggle.com/crowdflower/twitter-airline-sentiment

@author: asarker

'''

import nltk
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import *
from nltk.corpus import stopwords
import numpy as np

st = stopwords.words('english')
stemmer = PorterStemmer()

word_clusters = {}

def loadwordclusters():
    infile = open('./50mpaths2')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)


def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df


def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # Replace/remove username
    # raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    #stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))


if __name__ == '__main__':
    #Load the data
    f_path = './AirlineSentiment.csv'
    data = loadDataAsDataFrame(f_path)
    texts = data['text']
    classes = data['airline_sentiment']
    ids = data['tweet_id']

    word_clusters = loadwordclusters()

    #SPLIT THE DATA (we could use sklearn.model_selection.train_test_split)
    training_set_size = int(0.8*len(data))
    training_data = data[:training_set_size]
    training_texts = texts[:training_set_size]
    training_classes = classes[:training_set_size]
    training_ids = ids[:training_set_size]

    test_data = data[training_set_size:]
    test_texts = texts[training_set_size:]
    test_classes = classes[training_set_size:]
    test_ids = ids[training_set_size:]

    #PREPROCESS THE DATA
    training_texts_preprocessed = []
    test_texts_preprocessed = []
    test_clusters = []
    training_clusters = []

    #PROGRAMMING TIP: c++ style coding here can help when doing feature engineering.. see below
    training_texts_preprocessed = []
    for tr in training_texts:
        # you can do more with the training text here and generate more features...
        training_texts_preprocessed.append(preprocess_text(tr))
        training_clusters.append(getclusterfeatures(tr))
    for tt in test_texts:
        test_texts_preprocessed.append(preprocess_text(tt))
        test_clusters.append(getclusterfeatures(tt))
        #print(getclusterfeatures(tt))
    #VECTORIZE
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000)
    clustervectorizer = CountVectorizer(ngram_range=(1,3), max_features=10000)

    training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed).toarray()
    test_data_vectors = vectorizer.transform(test_texts_preprocessed).toarray()

    training_cluster_vectors = clustervectorizer.fit_transform(training_clusters).toarray()
    test_cluster_vectors = clustervectorizer.transform(test_clusters).toarray()

    training_data_vectors = np.concatenate((training_data_vectors, training_cluster_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors,test_cluster_vectors),axis=1)

    #TRAIN THE MODEL
    #gnb_classifier = GaussianNB()
    svm_classifier = svm.SVC(C=1, cache_size=200,
                             coef0=0.0, degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=True,
                             random_state=None, shrinking=True, tol=0.001, verbose=False)


    classifier = svm_classifier.fit(training_data_vectors, training_classes)


    # EVALUATE
    predictions = svm_classifier.predict(test_data_vectors)
    from sklearn.metrics import accuracy_score
    print (accuracy_score(predictions,test_classes))
'''
0.7127732240437158

'''