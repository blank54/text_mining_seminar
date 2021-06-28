##µ¥ÀÌÅÍ: ³×ÀÌ¹ö ¿µÈ­ ¸®ºä µ¥ÀÌÅÍ
##ÃÑ 200,000°³·Î ±¸¼ºµÈ µ¥ÀÌÅÍ·Î, 
##¸®ºä ÅØ½ºÆ®°¡ ±àÁ¤ÀÎ °æ¿ì 1, ºÎÁ¤ÀÎ °æ¿ì 0 À¸·Î Ç¥½ÃÇÑ ·¹ÀÌºí·Î ±¸¼º

#https://wikidocs.net/85337
#https://wikidocs.net/44249

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#µ¥ÀÌÅÍ ·Îµå
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")


train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')


print('ÈÆ·Ã¿ë ¸®ºä °³¼ö :',len(train_data)) # ÈÆ·Ã¿ë ¸®ºä °³¼ö Ãâ·Â
train_data[0:20] 

print('Å×½ºÆ®¿ë ¸®ºä °³¼ö :',len(test_data)) # Å×½ºÆ®¿ë ¸®ºä °³¼ö Ãâ·Â
test_data[:10]

#µ¥ÀÌÅÍ Á¤Á¦
#train_data Áßº¹ ÀÖ´ÂÁö È®ÀÎ
train_data['document'].nunique(), train_data['label'].nunique()

# document ¿­¿¡¼­ Áßº¹ÀÎ ³»¿ëÀÌ ÀÖ´Ù¸é Áßº¹ Á¦°Å
train_data.drop_duplicates(subset=['document'], inplace=True) 
print('ÃÑ »ùÇÃÀÇ ¼ö :',len(train_data))

#train_data¿¡¼­ ÇØ´ç ¸®ºäÀÇ ±à, ºÎÁ¤ À¯¹«°¡ ±âÀçµÇ¾îÀÖ´Â ·¹ÀÌºí(label) °ªÀÇ ºĞÆ÷
train_data['label'].value_counts().plot(kind = 'bar')
print(train_data.groupby('label').size().reset_index(name = 'count'))


#¸®ºä Áß¿¡ Null °ªÀ» °¡Áø »ùÇÃÀÌ ÀÖ´ÂÁö È®ÀÎ
print(train_data.isnull().values.any())
# Null °ªÀ» °¡Áø »ùÇÃÀÌ ¾î¶² ¿­¿¡ Á¸ÀçÇÏ´ÂÁö È®ÀÎ
print(train_data.isnull().sum())
# Null °ªÀ» °¡Áø »ùÇÃÀÌ ¾î´À ÀÎµ¦½ºÀÇ À§Ä¡¿¡ Á¸ÀçÇÏ´ÂÁö Ãâ·Â
train_data.loc[train_data.document.isnull()]

# Null °ªÀ» °¡Áø »ùÇÃÀ» Á¦°Å
train_data = train_data.dropna(how = 'any') # Null °ªÀÌ Á¸ÀçÇÏ´Â Çà Á¦°Å
print(train_data.isnull().values.any()) # Null °ªÀÌ Á¸ÀçÇÏ´ÂÁö È®ÀÎ

print(len(train_data))


#µ¥ÀÌÅÍ ÀüÃ³¸®
train_data['document'] = train_data['document'].str.replace("[^¤¡-¤¾¤¿-¤Ó°¡-ÆR ]","")
# ÇÑ±Û°ú °ø¹éÀ» Á¦¿ÜÇÏ°í ¸ğµÎ Á¦°Å
train_data[:20]

# train_data¿¡ °ø¹é(white space)¸¸ ÀÖ°Å³ª ºó °ªÀ» °¡Áø ÇàÀÌ ÀÖ´Ù¸é Null °ªÀ¸·Î º¯°æÇÏµµ·Ï
#ÇÑ ¹ø Null °ªÀÌ Á¸ÀçÇÏ´ÂÁö È®ÀÎ
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space µ¥ÀÌÅÍ¸¦ empty value·Î º¯°æ
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

#Null °ªÀÌ Á¸ÀçÇÏ´Â Çà Ãâ·Â
train_data.loc[train_data.document.isnull()][:5]

# Null°ª °¡Áø µ¥ÀÌÅÍ Á¦°Å
train_data = train_data.dropna(how = 'any')
print(len(train_data))

#Å×½ºÆ® µ¥ÀÌÅÍ¿¡ Áö±İ±îÁö ÁøÇàÇß´ø ÀüÃ³¸® °úÁ¤µéÀ» µ¿ÀÏÇÏ°Ô ÁøÇà
test_data.drop_duplicates(subset = ['document'], inplace=True) # document ¿­¿¡¼­ Áßº¹ÀÎ ³»¿ëÀÌ ÀÖ´Ù¸é Áßº¹ Á¦°Å
test_data['document'] = test_data['document'].str.replace("[^¤¡-¤¾¤¿-¤Ó°¡-ÆR ]","") # Á¤±Ô Ç¥Çö½Ä ¼öÇà
test_data['document'] = test_data['document'].str.replace('^ +', "") # °ø¹éÀº empty °ªÀ¸·Î º¯°æ
test_data['document'].replace('', np.nan, inplace=True) # °ø¹éÀº Null °ªÀ¸·Î º¯°æ
test_data = test_data.dropna(how='any') # Null °ª Á¦°Å
print('ÀüÃ³¸® ÈÄ Å×½ºÆ®¿ë »ùÇÃÀÇ °³¼ö :',len(test_data))


#ÅäÅ«È­
from konlpy.tag import Okt  
stopwords = ['ÀÇ','°¡','ÀÌ','Àº','µé','´Â','Á»','Àß','°Á','°ú','µµ','¸¦','À¸·Î','ÀÚ','¿¡','¿Í','ÇÑ','ÇÏ´Ù']

X_train = []
okt = Okt()
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # ÅäÅ«È­
    temp_X = [word for word in temp_X if not word in stopwords] # ºÒ¿ë¾î Á¦°Å
    X_train.append(temp_X)
    
print(X_train[:3])

X_test = []
for sentence in test_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # ÅäÅ«È­
    temp_X = [word for word in temp_X if not word in stopwords] # ºÒ¿ë¾î Á¦°Å
    X_test.append(temp_X)
    
print(X_test[:3])




# Á¤¼ö ÀÎÄÚµù
#ÈÆ·Ã µ¥ÀÌÅÍ¿¡ ´ëÇØ¼­ ´Ü¾î ÁıÇÕ(vocaburary) »ı»ó
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

#´Ü¾î ÁıÇÕÀÌ »ı¼ºµÇ´Â µ¿½Ã¿¡ °¢ ´Ü¾î¿¡ °íÀ¯ÇÑ Á¤¼ö°¡ ºÎ¿©µÊ
#tokenizer.word_index¸¦ Ãâ·ÂÇÏ¿© È®ÀÎ °¡´É
print(tokenizer.word_index)


#µîÀå ºóµµ¼ö°¡ 3È¸ ¹Ì¸¸(2È¸ ÀÌÇÏ)ÀÎ ´Ü¾îµéÀÌ ÀÌ µ¥ÀÌÅÍ¿¡¼­ ¾ó¸¸Å­ ºñÁßÀ» Â÷ÁöÇÏ´ÂÁö È®ÀÎ
threshold = 3
total_cnt = len(tokenizer.word_index) # ´Ü¾îÀÇ ¼ö
rare_cnt = 0 # µîÀå ºóµµ¼ö°¡ thresholdº¸´Ù ÀÛÀº ´Ü¾îÀÇ °³¼ö¸¦ Ä«¿îÆ®
total_freq = 0 # ÈÆ·Ã µ¥ÀÌÅÍÀÇ ÀüÃ¼ ´Ü¾î ºóµµ¼ö ÃÑ ÇÕ
rare_freq = 0 # µîÀå ºóµµ¼ö°¡ thresholdº¸´Ù ÀÛÀº ´Ü¾îÀÇ µîÀå ºóµµ¼öÀÇ ÃÑ ÇÕ
# ´Ü¾î¿Í ºóµµ¼öÀÇ ½Ö(pair)À» key¿Í value·Î ¹Ş´Â´Ù.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    # ´Ü¾îÀÇ µîÀå ºóµµ¼ö°¡ thresholdº¸´Ù ÀÛÀ¸¸é
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
print('´Ü¾î ÁıÇÕ(vocabulary)ÀÇ Å©±â :',total_cnt)
print('µîÀå ºóµµ°¡ %s¹ø ÀÌÇÏÀÎ Èñ±Í ´Ü¾îÀÇ ¼ö: %s'%(threshold - 1, rare_cnt))
print("´Ü¾î ÁıÇÕ¿¡¼­ Èñ±Í ´Ü¾îÀÇ ºñÀ²:", (rare_cnt / total_cnt)*100)
print("ÀüÃ¼ µîÀå ºóµµ¿¡¼­ Èñ±Í ´Ü¾î µîÀå ºóµµ ºñÀ²:", (rare_freq / total_freq)*100)


# ÀüÃ¼ ´Ü¾î °³¼ö Áß ºóµµ¼ö 2ÀÌÇÏÀÎ ´Ü¾î´Â Á¦°Å.
# 0¹ø ÆĞµù ÅäÅ«À» °í·ÁÇÏ¿© + 1
vocab_size = total_cnt - rare_cnt + 1
print('´Ü¾î ÁıÇÕÀÇ Å©±â :',vocab_size)


#´Ü¾î ÁıÇÕÀ» ÄÉ¶ó½º ÅäÅ©³ªÀÌÀúÀÇ ÀÎÀÚ·Î ³Ñ°ÜÁÖ¸é, 
#ÄÉ¶ó½º ÅäÅ©³ªÀÌÀú´Â ÅØ½ºÆ® ½ÃÄö½º¸¦ ¼ıÀÚ ½ÃÄö½º·Î º¯È¯
tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

#Á¤¼ö ÀÎÄÚµù °á°ú È®ÀÎ
print(X_train[:3])



#train_data¿¡¼­ y_train°ú y_test¸¦ º°µµ·Î ÀúÀå
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])


#ºó »ùÇÃ(empty samples) Á¦°Å
#±æÀÌ°¡ 0ÀÎ »ùÇÃµéÀÇ ÀÎµ¦½º¸¦ ¹ŞÀ½
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]


# ºó »ùÇÃµéÀ» Á¦°Å
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))


#ÆĞµù#¼­·Î ´Ù¸¥ ±æÀÌÀÇ »ùÇÃµéÀÇ ±æÀÌ¸¦ µ¿ÀÏÇÏ°Ô ¸ÂÃç
print('¸®ºäÀÇ ÃÖ´ë ±æÀÌ :',max(len(l) for l in X_train))
print('¸®ºäÀÇ Æò±Õ ±æÀÌ :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('ÀüÃ¼ »ùÇÃ Áß ±æÀÌ°¡ %s ÀÌÇÏÀÎ »ùÇÃÀÇ ºñÀ²: %s'%(max_len, (cnt / len(nested_list))*100))
  
max_len = 30
below_threshold_len(max_len, X_train)


#¸ğµç »ùÇÃÀÇ ±æÀÌ¸¦ 30À¸·Î ¸ÂÃß
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)


#³×ÀÌ¹ö ¿µÈ­ ¸®ºä ºĞ·ù
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


#ÇÏÀÌÆÛ ÆÄ¶ó¹ÌÅÍ Á¤ÀÇ
embedding_dim = 128
dropout_prob = (0.5, 0.8)
num_filters = 128


#ÀÔ·Â Ãş°ú ÀÓº£µù ÃşÀ» Á¤ÀÇ
model_input = Input(shape = (max_len,))
z = Embedding(vocab_size, embedding_dim, input_length = max_len, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)




#3, 4, 5ÀÇ Å©±â¸¦ °¡Áö´Â Ä¿³ÎÀ» °¢°¢ 128°³ »ç¿ë
# ÀÌµéÀ» maxpooling
conv_blocks = []

for sz in [3, 4, 5]:
    conv = Conv1D(filters = num_filters,
                         kernel_size = sz,
                         padding = "valid",
                         activation = "relu",
                         strides = 1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
    
    
    
    
#°¢°¢ maxpoolingÇÑ °á°ú¸¦ ¿¬°á(concatenate)
# ÀÌµé ¹ĞÁıÃş(dense layer)À¸·Î ¿¬°á
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(128, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])


#ÀÌÁø ºĞ·ù¸¦ ¼öÇà
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('CNN_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, y_train, batch_size = 64, epochs=10, validation_data = (X_test, y_test), verbose=2, callbacks=[es, mc])



#ÇĞ½À ÈÄ¿¡´Â ÀúÀåÇÑ ¸ğµ¨À» ·ÎµåÇÏ¿© Å×½ºÆ® µ¥ÀÌÅÍ¿¡ ´ëÇØ¼­ Æò°¡
loaded_model = load_model('CNN_model.h5')
print("\n Å×½ºÆ® Á¤È®µµ: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))



#¸®ºä ¿¹ÃøÇØº¸±â
def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # ÅäÅ«È­
  new_sentence = [word for word in new_sentence if not word in stopwords] # ºÒ¿ë¾î Á¦°Å
  encoded = tokenizer.texts_to_sequences([new_sentence]) # Á¤¼ö ÀÎÄÚµù
  pad_new = pad_sequences(encoded, maxlen = max_len) # ÆĞµù
  score = float(model.predict(pad_new)) # ¿¹Ãø
  if(score > 0.5):
    print("{:.2f}% È®·ü·Î ±àÁ¤ ¸®ºäÀÔ´Ï´Ù.\n".format(score * 100))
  else:
    print("{:.2f}% È®·ü·Î ºÎÁ¤ ¸®ºäÀÔ´Ï´Ù.\n".format((1 - score) * 100))
    
    

sentiment_predict('ÀÌ ¿µÈ­ °³²ÜÀë ¤»¤»¤»')
sentiment_predict('ÀÌ ¿µÈ­ ÇÙ³ëÀë ¤Ğ¤Ğ')
sentiment_predict('ÀÌµı°Ô ¿µÈ­³Ä ¤¹¤¹')




  
  
  








