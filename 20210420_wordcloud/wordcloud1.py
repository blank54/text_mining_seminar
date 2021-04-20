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
    #stopwords.add('장관') 
    #stopwords.add('것으로') 
    
    
    wordcloud = WordCloud(font_path='font/NanumGothic.ttf', background_color='white',stopwords=stopwords).generate(a)
    
    
    
    
plt.figure(figsize=(22,22)) #이미지 사이즈 지정
plt.imshow(wordcloud, interpolation='lanczos') #이미지의 부드럽기 정도
plt.axis('off') #x y 축 숫자 제거
plt.show() 
    