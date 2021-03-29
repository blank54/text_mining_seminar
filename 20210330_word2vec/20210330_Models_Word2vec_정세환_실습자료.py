#!/usr/bin/env python
# coding: utf-8

# # Configuration
import gensim
import nltk
import json
from glob import glob
import logging
from nltk.tokenize import word_tokenize
from pprint import pprint # pretty print | https://docs.python.org/ko/3/library/pprint.html

nltk.download('punkt')

def load_json_corpus(corpus_dir):
    fpaths = glob(corpus_dir + '/*')
    corpus = []
    for path in fpaths:
        with open(path, 'r') as f:
            doc = json.load(f)
            content = doc['content']
            doc_text = word_tokenize(content)
            corpus.append(doc_text)
    
    return corpus

import gensim.downloader as api
""" It will take a few minutes """
google_w2v = api.load('word2vec-google-news-300')

# Get word vector
def word_vector(query, model):
    if isinstance(model, gensim.models.word2vec.Word2Vec):
        result = model.wv[query]
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        result = model[query]
    else:
        print('No Word2vec model was provided.')
    
    return result

pprint(word_vector('world', google_w2v)[:10])
pprint(word_vector('bank', google_w2v)[:10])


# Similarity b/w 2 words
def word_similarity(query1, query2, model):
    from scipy.spatial.distance import cosine
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
    wv1 = word_vector(query1, model)
    wv2 = word_vector(query2, model)
    sim = 1 - cosine(wv1, wv2)
    return sim

query1 = 'world'
query2 = 'bank'
print(word_similarity(query1, query2, google_w2v))


# Get most similar words
def similar_words(query, model, k):
    if isinstance(model, gensim.models.word2vec.Word2Vec):
        return model.wv.most_similar(query, topn=k)
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        return model.most_similar(query, topn=k)
    else:
        print('No Word2vec model was provided.')

query = 'world'
k = 15
print('Most %d similar words with "%s"' % (k, query))
pprint(similar_words(query, google_w2v, k))


# Additive Composition
def add_comp(pos1, neg1, pos2, model, k):
    """
    Usage:
    Positive word 1 - Negative word 1 + Positive word 2 = Result
    Same as Pos1 : Neg1 = Result : Pos2
    (e.g. "Korea" - "Seoul" + "Tokyo" = ? ; i.e. Korea:Seoul = ?:Tokyo)
    """
    if isinstance(model, gensim.models.word2vec.Word2Vec):
        res = model.wv.most_similar(positive=[pos1, pos2], negative=[neg1], topn=k)
    elif isinstance(model, gensim.models.keyedvectors.KeyedVectors):
        res = model.most_similar(positive=[pos1, pos2], negative=[neg1], topn=k)
    else:
        print('No Word2vec model was provided.')
        res = None
    
    return res

pos1 = 'Korea'
neg1 = 'Seoul'
pos2 = 'Tokyo'
# pos1 : neg1 = (result) : pos2
k = 5
print('%d candidate words for the nation whose capital city is %s:' % (k, pos2))
pprint(add_comp(pos1, neg1, pos2, google_w2v, k)) # Expecting "Japan"
