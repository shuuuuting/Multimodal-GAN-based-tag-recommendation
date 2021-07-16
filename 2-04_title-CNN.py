#%%
from gensim.models import word2vec
w2vmodel = word2vec.Word2Vec.load('word2vec_wiki.model')

#%%
from keras.preprocessing.sequence import pad_sequences
title = pd.DataFrame(list(postsdb.find({},{"_id": 0,"postid": 1,"title": 1})))
title = title.set_index('postid')
embedding_title_train = []
i = 0
for tr_idx in train_id:
    embedding_title_train.append([])
    for word in title.title[tr_idx]:
        try:
            embedding_title_train[i].append(w2vmodel.wv[word])
        except:
            embedding_title_train[i].append(np.zeros(200))
    i+=1
embedding_title_train = pad_sequences(embedding_title_train,dtype='float32',maxlen=10) #(_,10,200)

embedding_title_test = []
i = 0
for te_idx in test_id:
    embedding_title_test.append([])
    for word in title.title[te_idx]:
        try:
            embedding_title_test[i].append(w2vmodel.wv[word])
        except:
            embedding_title_test[i].append(np.zeros(200))
    i+=1
embedding_title_test = pad_sequences(embedding_title_test,dtype='float32',maxlen=10)

#%%
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import MaxPooling1D,Conv1D
from keras.layers import Input, Dense, Embedding, Dropout
from keras.optimizers import Adam

embedding_size = 200 
title_length = 10
TopK = 10
batch_size = 64
optimizer = Adam(lr=0.0005, decay=0.000001)
num_tags = tag_train.shape[1]
def titlecnn():
    inp = Input(shape=(title_length,embedding_size))
    conv1 = Conv1D(embedding_size, 3, padding='same', activation='relu')(inp) 
    conv2 = Conv1D(embedding_size, 4, padding='same', activation='relu')(inp)
    conv3 = Conv1D(embedding_size, 5, padding='same', activation='relu')(inp)
    cnn = Add(name='ecnn')([conv1, conv2, conv3])
    cnn = SeqSelfAttention(attention_activation='sigmoid')(cnn)
    cnn.set_shape((cnn.shape[0],title_length,embedding_size))
    #cnn = MaxPooling1D()(cnn)
    #cnn = concatenate([conv1, conv2, conv3], axis=-1)
    flat = Flatten()(cnn)
    #flat = GlobalAveragePooling1D()(cnn)
    #x = Dense(1000, activation='relu')(flat)
    #x = Dropout(0.1)(x)
    #x = Dense(500, activation='relu')(flat)
    x = Dropout(0.2)(flat)
    x = Dense(num_tags, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = titlecnn()
model.fit(embedding_title_train, tag_train.astype(np.float32), epochs=50, 
    batch_size=batch_size, validation_data=(embedding_title_test, tag_test), callbacks=[early_stopping])
model.save('pretrained_eCNN.h5')
y_pred = model.predict(embedding_title_test)
acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)

##### word2vec pretrain by wiki #####
#%%
#acc:  0.6287477954144621
#precision:  0.11499118165784833
#recall:  0.4875832843425436
#f1:  0.17541456677880254