#!/usr/bin/env python
# coding: utf-8

# # 1. TF-IDF Embedding from Scratch

import glob, json
import numpy as np


# ## 1. 코퍼스 구축
file_paths = glob.glob('./WorldBankNews/*')

corpus = []
for path in file_paths:
  with open(path) as json_file:
    json_data = json.load(json_file)
  corpus.append(json_data)

import os
os.listdir('./WorldBankNews/')

texts = []

for doc in corpus:
  texts.append(doc['content'])


# ## 2. 토큰화 및 단어사전 구축
texts_tokens = []

documents_tokens = []
for text in texts:
  text_tokens = text.split()
  texts_tokens.append(text_tokens)

vocabulary = set()
for text_tokens in texts_tokens:
  vocab_in_doc = set(text_tokens)
  vocabulary.update(vocab_in_doc)

vocabulary = list(vocabulary)


# ## 3. Term-document Matrix (tf)
V = len(vocabulary)
D = len(corpus)

tf = np.zeros((V, D))
for j, text_tokens in enumerate(texts_tokens):
  for token in text_tokens:
    i = vocabulary.index(token)
    
    tf[i, j] += 1

print('term-document matrix (term frequency matrix)')
print(tf)


# ## 4. idf
df = np.count_nonzero(tf, axis=1, keepdims=True)
idf = D / df
tf_idf = tf * idf

print()
print('tf-idf matrix')
print(tf_idf)


# ## 6. 번외: 검색 기능 구현
print()
print('tf-idf document search demo:')
def cos_sim(vec1, vec2):
  from numpy.linalg import norm
  from numpy import dot
  
  result = dot(vec1, vec2) / (norm(vec1)*norm(vec2))
  
  return result

def tf_search(query):
  # query = list of tokens
  search_result = []

  query_tf = np.zeros(V)
  for token in query:
    i = vocabulary.index(token)
    query_tf[i] += 1
  
  for j in range(D):
    sim = cos_sim(query_tf, tf[:, j]) ###
    title = corpus[j]['title']
    search_result.append((title, sim))

  search_result.sort(key=lambda tup: tup[1], reverse=True)

  return search_result

def tfidf_search(query):
  # query = list of tokens
  search_result = []

  query_tf = np.zeros(V)
  for token in query:
    i = vocabulary.index(token)
    query_tf[i] += 1
  
  for j in range(D):
    sim = cos_sim(query_tf, tf_idf[:, j]) ###
    title = corpus[j]['title']
    search_result.append((title, sim))

  search_result.sort(key=lambda tup: tup[1], reverse=True)

  return search_result

query = ['a', 'an', 'the', 'and', 'with', 'of', 'Moldova']
print(query)

print(tf_search(query)[:2])
print(tfidf_search(query)[:2])


# # 2. TF-IDF Embedding Using Scikit-learn
print()
print('TF-IDF Embedding Using Scikit-learn')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

cv = CountVectorizer()
tf = cv.fit_transform(texts)

# Vocabulary
vocabulary = cv.get_feature_names()
print(vocabulary[:10])
print(vocabulary[-10:])

tt = TfidfTransformer()
tt.fit(tf)
tf_idf = tt.transform(tf)

print()
print('tf matrix')
print(tf_idf.shape)
print(tf_idf.toarray())

tv = TfidfVectorizer()
tf_idf_2 = tv.fit_transform(texts)

print()
print('tf-idf matrix')
print(tf_idf_2.toarray())


"""import random
len_query = 5
query = random.sample(vocabulary, len_query)
query.append('Moldova')
print(query)


# In[ ]:


query_string = ' '.join(query)
query_tf = cv.transform([query_string])


# In[ ]:


query_tf


# In[ ]:


query_vec = query_tf.toarray().flatten()
tf_idf = tf_idf.toarray()


# In[ ]:


query_vec.shape


# In[ ]:


def tfidf_search(query):
  # query = list of tokens
  search_result = []
  
  for i in range(D):
    sim = cos_sim(query, tf_idf[i, :]) ###
    title = corpus[i]['title']
    search_result.append((title, sim))

  search_result.sort(key=lambda tup: tup[1], reverse=True)

  return search_result


# In[ ]:


tfidf_search(query_vec)

"""