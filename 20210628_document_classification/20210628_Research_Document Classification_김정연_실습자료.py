##������: ���̹� ��ȭ ���� ������
##�� 200,000���� ������ �����ͷ�, 
##���� �ؽ�Ʈ�� ������ ��� 1, ������ ��� 0 ���� ǥ���� ���̺�� ����

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

#������ �ε�
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")


train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')


print('�Ʒÿ� ���� ���� :',len(train_data)) # �Ʒÿ� ���� ���� ���
train_data[0:20] 

print('�׽�Ʈ�� ���� ���� :',len(test_data)) # �׽�Ʈ�� ���� ���� ���
test_data[:10]

#������ ����
#train_data �ߺ� �ִ��� Ȯ��
train_data['document'].nunique(), train_data['label'].nunique()

# document ������ �ߺ��� ������ �ִٸ� �ߺ� ����
train_data.drop_duplicates(subset=['document'], inplace=True) 
print('�� ������ �� :',len(train_data))

#train_data���� �ش� ������ ��, ���� ������ ����Ǿ��ִ� ���̺�(label) ���� ����
train_data['label'].value_counts().plot(kind = 'bar')
print(train_data.groupby('label').size().reset_index(name = 'count'))


#���� �߿� Null ���� ���� ������ �ִ��� Ȯ��
print(train_data.isnull().values.any())
# Null ���� ���� ������ � ���� �����ϴ��� Ȯ��
print(train_data.isnull().sum())
# Null ���� ���� ������ ��� �ε����� ��ġ�� �����ϴ��� ���
train_data.loc[train_data.document.isnull()]

# Null ���� ���� ������ ����
train_data = train_data.dropna(how = 'any') # Null ���� �����ϴ� �� ����
print(train_data.isnull().values.any()) # Null ���� �����ϴ��� Ȯ��

print(len(train_data))


#������ ��ó��
train_data['document'] = train_data['document'].str.replace("[^��-����-�Ӱ�-�R ]","")
# �ѱ۰� ������ �����ϰ� ��� ����
train_data[:20]

# train_data�� ����(white space)�� �ְų� �� ���� ���� ���� �ִٸ� Null ������ �����ϵ���
#�� �� Null ���� �����ϴ��� Ȯ��
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space �����͸� empty value�� ����
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

#Null ���� �����ϴ� �� ���
train_data.loc[train_data.document.isnull()][:5]

# Null�� ���� ������ ����
train_data = train_data.dropna(how = 'any')
print(len(train_data))

#�׽�Ʈ �����Ϳ� ���ݱ��� �����ߴ� ��ó�� �������� �����ϰ� ����
test_data.drop_duplicates(subset = ['document'], inplace=True) # document ������ �ߺ��� ������ �ִٸ� �ߺ� ����
test_data['document'] = test_data['document'].str.replace("[^��-����-�Ӱ�-�R ]","") # ���� ǥ���� ����
test_data['document'] = test_data['document'].str.replace('^ +', "") # ������ empty ������ ����
test_data['document'].replace('', np.nan, inplace=True) # ������ Null ������ ����
test_data = test_data.dropna(how='any') # Null �� ����
print('��ó�� �� �׽�Ʈ�� ������ ���� :',len(test_data))


#��ūȭ
from konlpy.tag import Okt  
stopwords = ['��','��','��','��','��','��','��','��','��','��','��','��','����','��','��','��','��','�ϴ�']

X_train = []
okt = Okt()
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # ��ūȭ
    temp_X = [word for word in temp_X if not word in stopwords] # �ҿ�� ����
    X_train.append(temp_X)
    
print(X_train[:3])

X_test = []
for sentence in test_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # ��ūȭ
    temp_X = [word for word in temp_X if not word in stopwords] # �ҿ�� ����
    X_test.append(temp_X)
    
print(X_test[:3])




# ���� ���ڵ�
#�Ʒ� �����Ϳ� ���ؼ� �ܾ� ����(vocaburary) ����
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

#�ܾ� ������ �����Ǵ� ���ÿ� �� �ܾ ������ ������ �ο���
#tokenizer.word_index�� ����Ͽ� Ȯ�� ����
print(tokenizer.word_index)


#���� �󵵼��� 3ȸ �̸�(2ȸ ����)�� �ܾ���� �� �����Ϳ��� ��ŭ ������ �����ϴ��� Ȯ��
threshold = 3
total_cnt = len(tokenizer.word_index) # �ܾ��� ��
rare_cnt = 0 # ���� �󵵼��� threshold���� ���� �ܾ��� ������ ī��Ʈ
total_freq = 0 # �Ʒ� �������� ��ü �ܾ� �󵵼� �� ��
rare_freq = 0 # ���� �󵵼��� threshold���� ���� �ܾ��� ���� �󵵼��� �� ��
# �ܾ�� �󵵼��� ��(pair)�� key�� value�� �޴´�.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    # �ܾ��� ���� �󵵼��� threshold���� ������
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
print('�ܾ� ����(vocabulary)�� ũ�� :',total_cnt)
print('���� �󵵰� %s�� ������ ��� �ܾ��� ��: %s'%(threshold - 1, rare_cnt))
print("�ܾ� ���տ��� ��� �ܾ��� ����:", (rare_cnt / total_cnt)*100)
print("��ü ���� �󵵿��� ��� �ܾ� ���� �� ����:", (rare_freq / total_freq)*100)


# ��ü �ܾ� ���� �� �󵵼� 2������ �ܾ�� ����.
# 0�� �е� ��ū�� ����Ͽ� + 1
vocab_size = total_cnt - rare_cnt + 1
print('�ܾ� ������ ũ�� :',vocab_size)


#�ܾ� ������ �ɶ� ��ũ�������� ���ڷ� �Ѱ��ָ�, 
#�ɶ� ��ũ�������� �ؽ�Ʈ �������� ���� �������� ��ȯ
tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

#���� ���ڵ� ��� Ȯ��
print(X_train[:3])



#train_data���� y_train�� y_test�� ������ ����
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])


#�� ����(empty samples) ����
#���̰� 0�� ���õ��� �ε����� ����
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]


# �� ���õ��� ����
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))


#�е�#���� �ٸ� ������ ���õ��� ���̸� �����ϰ� ����
print('������ �ִ� ���� :',max(len(l) for l in X_train))
print('������ ��� ���� :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('��ü ���� �� ���̰� %s ������ ������ ����: %s'%(max_len, (cnt / len(nested_list))*100))
  
max_len = 30
below_threshold_len(max_len, X_train)


#��� ������ ���̸� 30���� ����
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)


#���̹� ��ȭ ���� �з�
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


#������ �Ķ���� ����
embedding_dim = 128
dropout_prob = (0.5, 0.8)
num_filters = 128


#�Է� ���� �Ӻ��� ���� ����
model_input = Input(shape = (max_len,))
z = Embedding(vocab_size, embedding_dim, input_length = max_len, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)




#3, 4, 5�� ũ�⸦ ������ Ŀ���� ���� 128�� ���
# �̵��� maxpooling
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
    
    
    
    
#���� maxpooling�� ����� ����(concatenate)
# �̵� ������(dense layer)���� ����
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(128, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])


#���� �з��� ����
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('CNN_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, y_train, batch_size = 64, epochs=10, validation_data = (X_test, y_test), verbose=2, callbacks=[es, mc])



#�н� �Ŀ��� ������ ���� �ε��Ͽ� �׽�Ʈ �����Ϳ� ���ؼ� ��
loaded_model = load_model('CNN_model.h5')
print("\n �׽�Ʈ ��Ȯ��: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))



#���� �����غ���
def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # ��ūȭ
  new_sentence = [word for word in new_sentence if not word in stopwords] # �ҿ�� ����
  encoded = tokenizer.texts_to_sequences([new_sentence]) # ���� ���ڵ�
  pad_new = pad_sequences(encoded, maxlen = max_len) # �е�
  score = float(model.predict(pad_new)) # ����
  if(score > 0.5):
    print("{:.2f}% Ȯ���� ���� �����Դϴ�.\n".format(score * 100))
  else:
    print("{:.2f}% Ȯ���� ���� �����Դϴ�.\n".format((1 - score) * 100))
    
    

sentiment_predict('�� ��ȭ ������ ������')
sentiment_predict('�� ��ȭ �ٳ��� �Ф�')
sentiment_predict('�̵��� ��ȭ�� ����')




  
  
  








