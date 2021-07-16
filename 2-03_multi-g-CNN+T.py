#每個群組各自學習一個CNN模型，最後只拿取CNN後的feature作為後續模型輸入
#%%
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import MaxPooling1D,Conv1D
from keras.layers import Input, Dense, Embedding, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D, Average, Add, concatenate
from keras.optimizers import Adam

embedding_size = 200 
seq_length = 50
title_length = 10
TopK = 10
batch_size = 64
optimizer = Adam(lr=0.0005, decay=0.000001)
num_tags = tag_train.shape[1]
def groupcnn():
    inp = Input(shape=(seq_length,embedding_size))
    inp_title = Input(shape=(title_length,embedding_size))
    inp_cnn = concatenate([inp, inp_title], axis=1)
    conv1 = Conv1D(embedding_size, 3, padding='same', activation='relu')(inp_cnn) 
    conv2 = Conv1D(embedding_size, 4, padding='same', activation='relu')(inp_cnn)
    conv3 = Conv1D(embedding_size, 5, padding='same', activation='relu')(inp_cnn)
    cnn = Add(name='grouptitlefeature')([conv1, conv2, conv3])
    cnn = SeqSelfAttention(attention_activation='sigmoid')(cnn)
    #cnn.set_shape((cnn.shape[0],seq_length,embedding_size))
    #cnn = MaxPooling1D()(cnn)
    #cnn = concatenate([conv1, conv2, conv3], axis=-1)
    #flat = Flatten()(cnn)
    flat = GlobalAveragePooling1D()(cnn)
    x = Dropout(0.2)(flat)
    #x = Dense(1000, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(500, activation='relu')(x)
    #x = Dropout(0.2)(x)
    x = Dense(num_tags, activation='sigmoid')(x)
    model = Model(inputs=[inp, inp_title], outputs=x)
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    return model

#%% each group topic run one CNN model
if os.path.exists(datadir+"grouptitleCNN_predict.txt"):
    os.remove(datadir+"grouptitleCNN_predict.txt")
f = open(datadir+"grouptitleCNN_predict.txt","w+")
global_group_title_train = {} #放CNN完後的feature, key是chanel number
global_group_title_test = {}
for i in range(n_clusters):
    local_trainid = np.where(np.array(groupid_train)==i)
    local_testid = np.where(np.array(groupid_test)==i)
    embedding_group_train = embedding_train[local_trainid]
    embedding_group_test = embedding_test[local_testid]
    embedding_g_title_train = embedding_title_train[local_trainid]
    embedding_g_title_test = embedding_title_test[local_testid]

    if len(embedding_group_test)==0:
        print('---------------------------------', file=f)
        print('no test example in group '+str(i), file=f)
        print('---------------------------------', file=f)
        global_group_title_train[i] = np.zeros((len(embedding_group_train),seq_length+title_length,embedding_size))
        continue
    local_tag_train = tag_train[local_trainid]
    local_tag_test = tag_test[local_testid]

    gCNN = groupcnn()
    gCNN.fit([embedding_group_train, embedding_g_title_train], local_tag_train, epochs=50, batch_size=4,
        validation_data=([embedding_group_test, embedding_g_title_test], local_tag_test), callbacks=[early_stopping])
    gExtraction = Model(inputs=gCNN.input, outputs=gCNN.get_layer('grouptitlefeature').output)
    y_pred = gCNN.predict([embedding_group_test, embedding_g_title_test])
    acc_K, precision_K, recall_K, f1_K = evaluation(local_tag_test, y_pred, TopK)
    print('---------------------------------', file=f)
    print('channel: %d'%i, file=f)
    print('number of trainset: %d'%len(embedding_group_train), file=f)
    print('number of testset: %d'%len(embedding_group_test), file=f)
    print('acc: %f'%acc_K, file=f)
    print('precision: %f'%precision_K, file=f)
    print('recall: %f'%recall_K, file=f)
    print('f1: %f'%f1_K, file=f)
    print('---------------------------------', file=f)
    global_group_title_train[i] = gExtraction([embedding_group_train, embedding_g_title_train])
    global_group_title_test[i] = gExtraction([embedding_group_test, embedding_g_title_test])
f.close()

#%%
group_title_train = {}
group_title_test = {} 
for i in global_group_title_train:
    group_title_train[i] = pd.DataFrame()
    group_title_train[i]['featurevec'] = list(np.array(global_group_title_train[i]))
    group_title_train[i].index = list(np.where(np.array(groupid_train)==i)) #index (順序非postid)
    if i not in global_group_title_test.keys(): continue
    group_title_test[i] = pd.DataFrame()
    group_title_test[i]['featurevec'] = list(np.array(global_group_title_test[i]))
    group_title_test[i].index = list(np.where(np.array(groupid_test)==i))

#%%
all_group_title_tr = pd.DataFrame()
for key, sub_df in group_title_train.items():
    all_group_title_tr = all_group_title_tr.append(sub_df, ignore_index=False) 
all_group_title_te = pd.DataFrame()
for key, sub_df in group_title_test.items():
    all_group_title_te = all_group_title_te.append(sub_df, ignore_index=False) 
all_group_title_tr = all_group_title_tr.sort_index()
all_group_title_te = all_group_title_te.sort_index()

group_title_tr_arr = np.array(all_group_title_tr.featurevec.tolist())
group_title_te_arr = np.array(all_group_title_te.featurevec.tolist())

#save arr
tr_arr_reshaped = group_title_tr_arr.reshape(group_title_tr_arr.shape[0], -1)
np.savetxt(datadir+"group_title_tr_arr.txt", tr_arr_reshaped)
te_arr_reshaped = group_title_te_arr.reshape(group_title_te_arr.shape[0], -1)
np.savetxt(datadir+"group_title_te_arr.txt", te_arr_reshaped)

#%%reload arr
loaded_arr = np.loadtxt(datadir+"group_title_tr_arr.txt")
group_title_tr_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"group_title_te_arr.txt")
group_title_te_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)

#%% each lda topic run one CNN model
if os.path.exists(datadir+"ldatitleCNN_predict.txt"):
    os.remove(datadir+"ldatitleCNN_predict.txt")
f = open(datadir+"ldatitleCNN_predict.txt","w+")
global_lda_title_train = {} #放CNN完後的feature, key是chanel number
global_lda_title_test = {}
for i in range(n_clusters):
    local_trainid = np.where(np.array(doc_topic)==i)
    local_testid = np.where(np.array(doc_topic_te)==i)
    embedding_lda_train = embedding_train[local_trainid]
    embedding_lda_test = embedding_test[local_testid]
    embedding_l_title_train = embedding_title_train[local_trainid]
    embedding_l_title_test = embedding_title_test[local_testid]

    if len(embedding_lda_test)==0:
        print('---------------------------------', file=f)
        print('no test example in topic '+str(i), file=f)
        print('---------------------------------', file=f)
        global_lda_title_train[i] = np.zeros((len(embedding_lda_train),seq_length+title_length,embedding_size))
        continue
    local_tag_train = tag_train[local_trainid]
    local_tag_test = tag_test[local_testid]

    gCNN = groupcnn()
    gCNN.fit([embedding_lda_train, embedding_l_title_train], local_tag_train, epochs=50, batch_size=4,
        validation_data=([embedding_lda_test, embedding_l_title_test], local_tag_test), callbacks=[early_stopping])
    gExtraction = Model(inputs=gCNN.input, outputs=gCNN.get_layer('grouptitlefeature').output)
    y_pred = gCNN.predict([embedding_lda_test, embedding_l_title_test])
    acc_K, precision_K, recall_K, f1_K, ndcg_K = evaluation(local_tag_test, y_pred, TopK)
    print('---------------------------------', file=f)
    print('channel: %d'%i, file=f)
    print('number of trainset: %d'%len(embedding_lda_train), file=f)
    print('number of testset: %d'%len(embedding_lda_test), file=f)
    print('acc: %f'%acc_K, file=f)
    print('precision: %f'%precision_K, file=f)
    print('recall: %f'%recall_K, file=f)
    print('f1: %f'%f1_K, file=f)
    print('ndcg: %f'%ndcg_K, file=f)
    print('---------------------------------', file=f)
    global_lda_title_train[i] = gExtraction([embedding_lda_train, embedding_l_title_train])
    global_lda_title_test[i] = gExtraction([embedding_lda_test, embedding_l_title_test])
f.close()

#%%
lda_title_train = {}
lda_title_test = {} 
for i in global_lda_title_train:
    lda_title_train[i] = pd.DataFrame()
    lda_title_train[i]['featurevec'] = list(np.array(global_lda_title_train[i]))
    lda_title_train[i].index = list(np.where(np.array(doc_topic)==i)) #index (順序非postid)
    if i not in global_lda_title_test.keys(): continue
    lda_title_test[i] = pd.DataFrame()
    lda_title_test[i]['featurevec'] = list(np.array(global_lda_title_test[i]))
    lda_title_test[i].index = list(np.where(np.array(doc_topic_te)==i))

#%%
all_lda_title_tr = pd.DataFrame()
for key, sub_df in lda_title_train.items():
    all_lda_title_tr = all_lda_title_tr.append(sub_df, ignore_index=False) 
all_lda_title_te = pd.DataFrame()
for key, sub_df in lda_title_test.items():
    all_lda_title_te = all_lda_title_te.append(sub_df, ignore_index=False) 
all_lda_title_tr = all_lda_title_tr.sort_index()
all_lda_title_te = all_lda_title_te.sort_index()

lda_title_tr_arr = np.array(all_lda_title_tr.featurevec.tolist())
lda_title_te_arr = np.array(all_lda_title_te.featurevec.tolist())

#save arr
tr_arr_reshaped = lda_title_tr_arr.reshape(lda_title_tr_arr.shape[0], -1)
np.savetxt(datadir+"lda_title_tr_arr.txt", tr_arr_reshaped)
te_arr_reshaped = lda_title_te_arr.reshape(lda_title_te_arr.shape[0], -1)
np.savetxt(datadir+"lda_title_te_arr.txt", te_arr_reshaped)

#%%reload arr
loaded_arr = np.loadtxt(datadir+"lda_title_tr_arr.txt")
lda_title_tr_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"lda_title_te_arr.txt")
lda_title_te_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
# %%
