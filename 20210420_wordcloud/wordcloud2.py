import konlpy.tag
from wordcloud import WordCloud,STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
import sys
from konlpy.tag import Okt
from collections import Counter


with open("/data/kjyon/crawling/wordcloud/forwc.txt", 'r', encoding='utf8') as f:
    content = f.read()
    
    
filtered_content = content.replace('.', '').replace(',','').replace("'","").replace('·', ' ').replace('=','').replace('\n','')


Okt = konlpy.tag.Okt()
Okt_morphs = Okt.pos(filtered_content)  # 튜플반환
print(Okt_morphs)


komoran = konlpy.tag.Komoran()
komoran_morphs = komoran.pos(filtered_content)
print(komoran_morphs)



Noun_words = []
for word, pos in Okt_morphs:
    if pos == 'Noun':
        Noun_words.append(word)
print(Noun_words)


stopwords = ['구독', '뉴스', '기자','무단','배포']
unique_Noun_words = set(Noun_words)
for word in unique_Noun_words:
    if word in stopwords:
        while word in Noun_words: Noun_words.remove(word)  # 최종결과 : Noun_words
        
        
        
from collections import Counter
c = Counter(Noun_words)
print(c.most_common(10)) # 상위 10개 출력하기



import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path

FONT_PATH = 'font/NanumGothic.ttf' # For Korean characters

noun_text = ''
for word in Noun_words:
    noun_text = noun_text +' '+word

wordcloud = WordCloud(max_font_size=60, relative_scaling=.5, font_path=FONT_PATH).generate(noun_text) # generate() 는 하나의 string value를 입력 받음
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


wordcloud = WordCloud(max_font_size=50, max_words=30, background_color='white', relative_scaling=.5, font_path=FONT_PATH).generate(noun_text)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()