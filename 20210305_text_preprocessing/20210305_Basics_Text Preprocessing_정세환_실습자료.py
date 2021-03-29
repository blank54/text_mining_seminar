#!/usr/bin/env python
# coding: utf-8

"""
Prerequisites:
NLTK
NLTK.download('stopwords')
NLTK.download('punkt')
NLTK.download('wordnet')
KoNLPy
"""

# # 0. 패키지 설치 및 불러오기

# 필요한 패키지는 모두 설치되어 있다고 가정
import nltk
import konlpy


# # 1. Stopword Removal

# 영어의 경우 NLTK가 자체적으로 가지고 있는 stopword list를 활용할 수 있습니다.
print('English stopword removal demo:')
from nltk.corpus import stopwords

# nltk.download('stopwords') # 필요한 경우 별도로 download
print(stopwords.words('english')[:10])

# print(stopwords.words('french')[:10])
# print(stopwords.words('korean')[:10])


# 이 리스트를 가지고 실제로 stopword removal을 해 보겠습니다. 토크나이징은 이미 되어 있다고 가정하고, 각각의 토큰 중 stopword에 해당하는 토큰은 제거해 보겠습니다.

example = ['Family', 'is', 'not', 'an', 'important', 'thing.', 'It', 'is', 'everything.']
print(example)

stop_words = set(stopwords.words('english'))
result = []

for w in example:
    if w not in stop_words:
        result.append(w)

print(result)


# Stopword에 해당하는 is, not, an, it이 제거된 것을 확인할 수 있습니다.

# 한국어 stopword list는 NLTK에 없기 때문에 따로 리스트를 만들어야 합니다.
print()
print('Korean stopword removal demo:')
# https://www.ranks.nl/stopwords/korean 에서 제안한 stopword list를 활용해 보겠습니다.

# f = open('./sample_data/korean_stopwords.txt', 'r')
f = open('./korean_stopwords.txt', 'r')
korean_stop_words = f.read()
f.close()

korean_stop_words = korean_stop_words.split('\n')

# 나머지 절차는 영어의 경우와 동일합니다.
from nltk.tokenize import word_tokenize
nltk.download('punkt')

example = '3월 4일 (목) 제 10회 연구실 워크숍 (치코리타) 가 예정되어있습니다. 오후 2시 ~ 6시 예정이며 최흥순 부장님, 황정빈 연구원, 장태연 연구원, 이규은 연구원 네분이 연사로 참여하실 예정입니다.'
word_tokens = word_tokenize(example)
print(word_tokens[:20])

result = []
for w in word_tokens:
    if w not in korean_stop_words:
        result.append(w)

print(result[:20])


# # 2. Stemming / Lemmatization

# Stemming과 Lemmatization은 주로 영어 텍스트의 전처리에 사용하는 방법입니다.
print()
print('Lemmatization demo:')
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
n = WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print(words)
print([n.lemmatize(w) for w in words])


# Lemmatization이 적용되는 단어의 품사를 미리 알 수 있다면 성능을 높일 수 있습니다.
# 아래 예시에서, dies라는 단어가 동사임을 알고 있다면 이의 원형은 die가 되야 한다는 것을 알 수 있습니다.

print('dies')
print(n.lemmatize('dies'))
print(n.lemmatize('dies', 'v'))
print()
print(n.lemmatize('watched'))
print(n.lemmatize('watched', 'v'))


# Stemmer중 (아마도) 가장 많이 사용되는 것은 Porter Stemmer입니다.
# https://tartarus.org/martin/PorterStemmer/index.html
print()
print('Stemming demo:')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
text = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words = word_tokenize(text)
print(words)
print([ps.stem(w) for w in words])


# 다른 하나는 Lancaster Stemmer입니다. Porter Stemmer와는 다른 규칙이 적용되고 결과도 다르게 나옵니다.
print()
print('comparison: Potter vs. Lancaster stemmer')
from nltk.stem import LancasterStemmer
ls = LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print(words)
print([ps.stem(w) for w in words])
print([ls.stem(w) for w in words])


# # 3. Tokenization / POS Tagging / Morpheme Analysis

# 토크나이징을 하는 가장 간단한 방법은 띄어쓰기 단위로 텍스트를 구분하는 것입니다.
print()
print('English tokenizing demo:')
example = "Time is an illusion. Lunchtime double so!"
result = example.split()
print(example)
print(result)


# 하지만 띄어쓰기가 반드시 토큰의 경계가 되는 것은 아닙니다.
from nltk.tokenize import word_tokenize
example = "Don't be fooled by the dark sounding name."
result = word_tokenize(example)
print()
print(example)
print(result)

# 반면에, 한국어 텍스트를 토크나이징하는 것은 영어처럼 쉽지 않습니다. 첫째, 한국어의 어절은 여러 유닛이 결합되어 있는 경우가 많으며, 둘째, 띄어쓰기가 제대로 지켜지지 않는 경우가 많기 때문입니다. 때문에, 한국어 토크나이징은 대부분 형태소 단위로 이루어지며, 이를 위해서 형태소 분석이 선행되어야 합니다. 경우에 따라서는 형태소 분석의 결과가 곧 토크나이징의 결과가 되기도 합니다.
# 
# KoNLPy는 Okt(Open Korea Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma) 총 네 가지 형태소 분석기를 제공합니다.
# 
# (참고로, Okt는 예전에는 Twitter 라는 이름을 갖고 있었으나 최신 버전에서 이름이 변경되었습니다.)
# 
# 각각의 형태소 분석기는 공통적으로 '형태소 추출', '품사 태깅', '명사 추출' 기능을 가지고 있습니다. 실습에서는 Okt와 꼬꼬마 두 개의 형태소 분석기의 토크나이징 결과를 비교해 보겠습니다.

print()
print('Korean tokenizing demo:')

example = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"
print('example sentence:')
print(example)

# Okt 형태소 분석기
print('Okt 형태소 분석기')
from konlpy.tag import Okt
okt = Okt()

print(okt.morphs(example))  # 형태소 추출
print(okt.pos(example))     # 품사 태깅
print(okt.nouns(example))   # 명사 추출

# 꼬꼬마 형태소 분석기
print('꼬꼬마 형태소 분석기')
from konlpy.tag import Kkma
kkma=Kkma()

print(kkma.morphs(example))
print(kkma.pos(example))  
print(kkma.nouns(example))  


# 하지만 미등록단어 (Out-of-vocabulary word) 가 많은 텍스트를 기존 형태소 분석기로 분석하면 여러 가지 문제가 발생합니다. 예를 들면, 건설분야의 전문용어가 많은 교량점검보고서의 다음 문장을 형태소 분석기로 분석해 보겠습니다.
# 
# > **국부적인 아스콘 패임과 전반적인 골재마모가 조사되었으며, 골재마모가 상대적으로 심한 구간에서는 포장체의 표면이 거칠거나 경미한 골재탈리가 진행되는 상태이다.**
# 
# 실제 점검보고서의 한 문장을 가져왔습니다. 아스콘, 골재마모(골재 + 마모), 골재탈리(골재+탈리)와 같이 전문용어 포함된 문장은 KoNLPy 토크나이저가 올바르게 분석하지 못하는 경우가 생깁니다.

example = '국부적인 아스콘 패임과 전반적인 골재마모가 조사되었으며, 골재마모가 상대적으로 심한 구간에서는 포장체의 표면이 거칠거나 경미한 골재탈리가 진행되는 상태이다.'
print()
print(example)
print(okt.pos(example))
print(kkma.pos(example))


# # # 4. Text Normalization

# # Text Normalization은 앞서 실습했던 stopword removal이나 stemming처럼 패키지로 구현된 경우가 흔하지 않습니다. 이는 데이터 분석가가 자신이 가지고 있는 텍스트 데이터의 특성에 따라 그때그때 normalization의 범위를 설정하고 규칙을 구현해야 하기 때문입니다. 예를 들어, 카카오톡 대화 텍스트에 자연어 처리 방법을 적용한다고 할 때, 어떤 normalization 방법이 적용되어야 할까요? 엄청나게 많은 전처리 과정이 필요하겠지만, 이번 실습에서는 딱 한 가지 규칙만 적용해 보겠습니다.

# # 카카오톡 대화 내용의 대부분을 차지하는 것은 "ㅋㅋㅋ"일 것입니다... 아마도. 그런데, 예를 들어 "ㅋㅋㅋㅋ"와 "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ" 사이에 차이가 있다고 볼 수 있을까요? 구체적으로는, "ㅋ"가 연달아 여러 개 나온 경우는 모두 한 가지 경우로 치환해서 볼 수 있겠습니다.

# # 이제 직접 normalization 규칙을 작성해 보겠습니다. 먼저, 이 규칙을 적용하게 되는 텍스트의 입력과 출력은 다음과 같다고 생각해 볼 수 있습니다.
# # ```
# # # input --> [Normalization 규칙] --> output
# # # 예1. ㅋㅋ --> [규칙] --> ㅋㅋ
# # # 예2. ㅋㅋㅋㅋ --> [규칙] --> ㅋㅋ
# # # 예3. ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ --> [규칙] --> ㅋㅋ
# # # 예4. ㅋ --> [규칙] --> Error. Not Applicable
# # ```
# # 몇 가지 예시로부터 보다 구체적인 규칙을 떠올릴 수 있습니다.
# # 
# # 
# # ```
# # # ㅋ이 두 개 이상 연달아 입력된 문자열이 입력되었을 때, ㅋㅋ를 출력한다.
# # # ㅋ이 한 개만 입력된 경우는 규칙이 적용되지 않는다.
# # ```
# # 이를 파이썬 코드로 구현하면 아래와 같습니다.
# # 
# # 

# # 이를 파이썬 코드로 구현하려면 몇 가지 함수를 알아야 합니다. 먼저, "ㅋ"라는 글자가 단어에 포함되어 있는 지 찾을 수 있어야 합니다. 이는 간단하게 
# # 
# # ```
# # # "ㅋ" in word
# # ```
# # 로 찾을 수 있습니다. 예를 들어,
# # 

# # In[28]:


# example = "ㅋㅋㅋㅋ"

# if "ㅋ" in example:
#   print("ㅋ가 포함된 문자열입니다.")
# else:
#   print("ㅋ가 포함되지 않은 문자열입니다.")


# # In[29]:


# example = "아무 문자열"

# if "ㅋ" in example:
#   print("ㅋ가 포함된 문자열입니다.")
# else:
#   print("ㅋ가 포함되지 않은 문자열입니다.")


# # 다음으로, ㅋ가 몇 개나 들어 있는지 세야 합니다. 이는 count()함수를 활용하면 가능합니다.

# # In[30]:


# example1 = "ㅋㅋ"
# example2 = "ㅋㅋㅋㅋㅋ"

# print(example1.count("ㅋ"))
# print(example2.count("ㅋ"))


# # In[32]:


# text = '메롱'
# result = ''
# num_k = text.count('ㅋ') # text 변수에서 ㅋ의 개수를 세서 num_k라는 변수에 저장

# if num_k >= 2: # ㅋ의 개수가 2개 이상이면
#   result = 'ㅋㅋ'
# else:
#   result = '규칙이 적용되지 않는 문자열입니다.'

# print(result)


# # 하지만 한 가지 문제가 있습니다. count 함수는 단순히 ㅋ의 개수만을 셀 뿐, ㅋ가 연달아 나왔는지는 고려하지 않습니다.

# # In[33]:


# text = '텍ㅋ스ㅋ트ㅋ마ㅋ이ㅋ닝ㅋ세ㅋ미ㅋ나'
# result = ''
# num_k = text.count('ㅋ') # text 변수에서 ㅋ의 개수를 세서 num_k라는 변수에 저장

# if num_k >= 2: # ㅋ의 개수가 2개 이상이면
#   result = 'ㅋㅋ'
# else:
#   result = '규칙이 적용되지 않는 문자열입니다.'

# print(result)


# # 규칙이 적용될 수 없는 문자열임에도 불구하고 ㅋ의 개수가 두 개 이상이기 때문에 'ㅋㅋ'를 출력하고 말았습니다. 즉, ㅋ가 여러 번 나왔을 뿐만 아니라 중간에 다른 문자가 끼지 않고 ㅋ만 연달아서 나오는 경우를 찾아낼 수 있어야 합니다.

# # 정규 표현식 (Regular Expression)은 이처럼 특정한 규칙을 가진 문자열을 다룰 수 있게 해 주는 언어 표현 방식 중 하나입니다. 정규 표현식은 파이썬 고유의 문법이나 패키지가 아니라 독자적으로 존재하는 규칙이고, 파이썬에서는 re 라는 이름의 패키지로 구현되어 있습니다.

# # In[34]:


# import re


# # 정규 표현식은 파이썬뿐만 아니라 다른 언어로도 자연어처리를 할 때 굉장히 유용하게 사용할 수 있습니다. 정규 표현식의 모든 것을 이번 실습에서 강의하기에는 시간이 모자라기 때문에, 이번에는 정규 표현식을 이용하면 전처리를 효율적으로 할 수 있다는 것만 보여주고 넘어가겠습니다. re 패키지의 자세한 사용 방법 설명은 생략하고, 특정 규칙이 어떻게 정규 표현식으로 구현되는지만 설명하도록 하겠습니다.

# # In[35]:


# p = re.compile('ㅋ{2,}') # 찾아내고 싶은 규칙을 정규 표현식으로 표현한 것


# # re.compile() 함수의 입력값 ㅋ{2,} 이 바로 앞서 세운 규칙을 정규 표현식으로 표현한 것입니다. 먼저, ㅋ라는 특정 글자만을 찾고 싶기 때문에 ㅋ를 검색 조건으로 넣습니다. 다음으로, ㅋ가 2번 이상 반복해서 나오는 경우를 찾고 싶을 땐, {} 표현을 사용하면 됩니다. ㅋ{m, n} 은 ㅋ가 m번 이상, n번 이하인 문자열을 찾으라는 것을 의미합니다. 우리가 찾고 싶은 규칙에는 상한선은 없으므로 {2,}로 썼습니다.

# # In[36]:


# text = 'ㅋㅋ'
# p.sub('ㅋㅋ', text) # text라는 문자열에서 규칙에 해당하는 부분을 'ㅋㅋ'라는 문자열로 치환 (substitute) 하는 함수


# # In[37]:


# example = 'ㅋㅋㅋㅋㅋㅋㅋ'
# p.sub('ㅋㅋ', example)


# # In[38]:


# example = 'ㅋ가 한 개만 있는 문자열'
# p.sub('ㅋㅋ', example)


# # In[39]:


# example = 'ㅋ가 여러 개 있는 ㅋㅋㅋㅋㅋㅋ 문자열ㅋㅋㅋㅋㅋㅋㅋㅋ'
# p.sub('ㅋㅋ', example)


# # ㅋ가 아니라 다른 문자, 예를 들면 ㅎ를 동일한 방식으로 전처리하고 싶다면, 정규 표현식의 규칙에서 ㅋ를 ㅎ로 바꿔주기만 하면 됩니다.

# # In[40]:


# example = 'ㅋ와 ㅎ가 섞여 있는 ㅋㅋㅋ 문자열에서 ㅎㅎㅎㅎㅎㅎ 만 ㅎㅎ로 바꾸려면?'
# p = re.compile('ㅎ{2,}')
# p.sub('ㅎㅎ', example)


# # 5. Vocabulary List

# 어휘 목록 (또는 단어 사전, 시소러스, 온톨로지... 뭐가 되었든) 을 구축하는 것은 프로그래밍 기술보다도 연구자의 insight와 인력 노가다가 훨씬 중요하다고 생각합니다. 즉, 자기가 가지고 있는 코퍼스에 어떤 어휘가 사용되고 있으며, 다른 어휘들과 어떤 관계인지 (유의어? 반의어? 상/하위어?), 어떤 개념으로 묶일 수 있는 지 (교량점검보고서의 경우, '부재 종류'에 해당하는 단어들이 어떤 게 있는 지) 등등... 이는 전처리 단계에서 코드 몇 줄로 뚝딱 끝낼 수 있는 문제는 아닙니다. 다만, 이번 실습에서는 단어 사전 구축의 가장 기본이라고 할 수 있는 '단어 개수 세기'를 간단히 해 보겠습니다.

# 토크나이징만 제대로 되어 있다고 가정하면 (=단어가 잘 분리되어 있다면), 단어 개수를 세는 것은 매우 간단합니다. NLTK에서는 Frequency distribution을 계산할 수 있는 FreqDist 함수가 구현되어 있습니다.

print()
print('NLTK word frequency count demo:')

from nltk.probability import FreqDist

example = "Let's return to our exploration of the ways we can bring our computational resources to bear on large quantities of text."
words = word_tokenize(example)
fdist = FreqDist(words)

print(example)
print(fdist.most_common(10))
