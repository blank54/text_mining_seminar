from wordcloud import WordCloud,STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
import sys
from konlpy.tag import Okt
from collections import Counter

df = pd.read_excel("/data/kjyon/crawling/wordcloud/result_combine_202012.xlsx")
df_sample = df['contents']



print(df_sample)



for i in range (20):
    text = []
    text.append(df_sample[i])
    i+=1
    #print(text)
    a= text[0]
    print(a)
    
    stopwords = set(STOPWORDS) 
    #stopwords.add('���') 
    #stopwords.add('������') 
    
    
    wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='white',stopwords=stopwords).generate(a)
    
    
    
    
plt.figure(figsize=(22,22)) #�̹��� ������ ����
plt.imshow(wordcloud, interpolation='lanczos') #�̹����� �ε巴�� ����
plt.axis('off') #x y �� ���� ����
plt.show() 
    