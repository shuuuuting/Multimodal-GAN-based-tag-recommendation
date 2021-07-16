#%%----------------------------------------------------------------------------------------
#如有照順序從第1大項重新跑的話，可以略過此區塊，不用從資料庫取資料
#------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import csv
from pymongo import MongoClient

conn = MongoClient('localhost', 27017) #連結mongodb
db = conn.NiusNews2020_04_12 #create database
text_train = pd.DataFrame(list(db['text_train_db'].find({},{"_id": 0,"postid": 1,"content_jieba": 1})))
text_test = pd.DataFrame(list(db['text_test_db'].find({},{"_id": 0,"postid": 1,"content_jieba": 1})))
tag_train = np.array(pd.DataFrame(list(db['tag_train_db'].find({},{"_id": 0}))))
tag_test = np.array(pd.DataFrame(list(db['tag_test_db'].find({},{"_id": 0}))))

##### keras word embedding #####
#%%----------------------------------------------------------------------------------------
#preprocess text for embedding layer
#------------------------------------------------------------------------------------------
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
text_train['sentence'] = [' '.join(map(str, l)) for l in text_train['content_jieba']]  
text_test['sentence'] = [' '.join(map(str, l)) for l in text_test['content_jieba']]  

#backup data
text_tr_pd = text_train
text_te_pd = text_test

doc_train = list(text_train['sentence'])
doc_test = list(text_test['sentence'])

#count no of vocs in corpus
from collections import Counter 
voc_list = [item for sublist in text_train['content_jieba'] for item in sublist]   
voc_list.extend([item for sublist in text_test['content_jieba'] for item in sublist])   
vocs = Counter(voc_list).keys() 
print("No of unique vocs are:", len(vocs)) #45807

#integer encode the documents
vocab_size = len(vocs)
text_train = [one_hot(d, vocab_size) for d in doc_train]
text_test = [one_hot(d, vocab_size) for d in doc_test]

#pad documents to a max length of 50 words
max_length = 50
text_train = pad_sequences(text_train, maxlen=max_length, padding='post') ##(_,50)
print("padded_docs_train:\n",text_train)
text_test = pad_sequences(text_test, maxlen=max_length, padding='post')

#%%----------------------------------------------------------------------------------------
#train text multi-label classification
#------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import MaxPooling1D,Conv1D
from keras.layers import Input, Dense, Embedding, Dropout, concatenate, Add, GlobalMaxPooling1D
from keras_self_attention import SeqSelfAttention
from keras.optimizers import Adam

embedding_size = 200 
seq_length = 50
TopK = 10
batch_size = 64
num_tags = tag_train.shape[1]
optimizer = Adam(lr=0.0005, decay=0.000001)

def textcnn():
    inp = Input(shape=(seq_length,), batch_size=batch_size)
    #x = embedding_layer(inp)
    x = Embedding(vocab_size, embedding_size, input_length=seq_length)(inp)
    conv1 = Conv1D(embedding_size, 3, padding='same', activation='relu')(x)
    #conv1 = SeqSelfAttention(attention_activation='sigmoid')(conv1)
    #conv1 = MaxPooling1D()(conv1)
    
    conv2 = Conv1D(embedding_size, 4, padding='same', activation='relu')(x)
    #conv2 = SeqSelfAttention(attention_activation='sigmoid')(conv2)
    #conv2 = MaxPooling1D()(conv2)
    
    conv3 = Conv1D(embedding_size, 5, padding='same', activation='relu')(x)
    #conv3 = SeqSelfAttention(attention_activation='sigmoid')(conv3)
    #conv3 = MaxPooling1D()(conv3)
    
    #cnn = concatenate([conv1, conv2, conv3], axis=-1)
    cnn = Add(name='tcnn')([conv1, conv2, conv3])
    cnn = SeqSelfAttention(attention_activation='sigmoid')(cnn)
    #cnn.set_shape((cnn.shape[0],seq_length,embedding_size))
    #cnn = AveragePooling1D(pool_size=seq_length//2, padding="same")(cnn) 
    #flat = GlobalMaxPooling1D()(cnn)
    flat = Flatten()(cnn)
    #x = Dropout(0.5)(flat)
    #x = Dense(1000, activation='relu')(x)
    #x = Dropout(0.75)(x)
    #x = Dense(500, activation='relu')(x)
    x = Dropout(0.2)(flat)
    x = Dense(num_tags, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
model = textcnn()
'''
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=seq_length))
model.add(Conv1D(embedding_size, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(600, activation='relu'))
model.add(Dense(num_tags, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
'''
#%%
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True) #持續5個epoch沒下降就停
model.fit(text_train, tag_train, epochs=24, batch_size=batch_size,
            validation_data=(text_test, tag_test.astype(np.float32)), 
            callbacks=[early_stopping])
y_pred = model.predict(text_test)
acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)
#451個tag
#acc:  0.5019638648860958
#precision:  0.09379418695993716
#recall:  0.35643493061010734
#f1:  0.14121133591236637

#263個tag
#acc:  0.6005291005291006
#precision:  0.10758377425044091
#recall:  0.4636033425715965
#f1:  0.16546407837646557


#%% 3kernel conv
#acc:  0.6428571428571429
#precision:  0.1215167548500882
#recall:  0.507695053329974
#f1:  0.18522101194366006

