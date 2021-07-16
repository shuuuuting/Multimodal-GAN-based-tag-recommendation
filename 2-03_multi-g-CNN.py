#每個群組各自學習一個CNN模型，最後只拿取CNN後的feature作為後續模型輸入
#%%
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import MaxPooling1D,Conv1D
from keras.layers import Input, Dense, Embedding, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D, Average, Add
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention

embedding_size = 200 
seq_length = 50
title_length = 10
TopK = 10
batch_size = 64
optimizer = Adam(lr=0.0005, decay=0.000001)
num_tags = tag_train.shape[1]
def groupcnn():
    inp = Input(shape=(seq_length,embedding_size))
    conv1 = Conv1D(embedding_size, 3, padding='same', activation='relu')(inp) 
    conv2 = Conv1D(embedding_size, 4, padding='same', activation='relu')(inp)
    conv3 = Conv1D(embedding_size, 5, padding='same', activation='relu')(inp)
    cnn = Add(name='groupfeature')([conv1, conv2, conv3])
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
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    return model

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True) #持續10個epoch沒下降就停

#%% each group topic run one CNN model
n_clusters = 15
if os.path.exists(datadir+"groupCNN_predict.txt"):
    os.remove(datadir+"groupCNN_predict.txt")
f = open(datadir+"groupCNN_predict.txt","w+")
global_group_train = {} #放CNN完後的feature, key是chanel number
global_group_test = {}
for i in range(n_clusters):
    local_trainid = np.where(np.array(groupid_train)==i)
    local_testid = np.where(np.array(groupid_test)==i)
    embedding_group_train = embedding_train[local_trainid]
    embedding_group_test = embedding_test[local_testid]

    if len(embedding_group_test)==0:
        print('---------------------------------', file=f)
        print('no test example in group '+str(i), file=f)
        print('---------------------------------', file=f)
        global_group_train[i] = np.zeros((len(embedding_group_train),50,200))
        continue
    local_tag_train = tag_train[local_trainid]
    local_tag_test = tag_test[local_testid]

    gCNN = groupcnn()
    gCNN.fit(embedding_group_train, local_tag_train, epochs=50, batch_size=4,
        validation_data=(embedding_group_test, local_tag_test), callbacks=[early_stopping])
    gExtraction = Model(inputs=gCNN.input, outputs=gCNN.get_layer('groupfeature').output)
    y_pred = gCNN.predict(embedding_group_test)
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
    global_group_train[i] = gExtraction(embedding_group_train)
    global_group_test[i] = gExtraction(embedding_group_test)
f.close()

#%%
group_train = {}
group_test = {} 
for i in global_group_train:
    group_train[i] = pd.DataFrame()
    group_train[i]['featurevec'] = list(np.array(global_group_train[i]))
    group_train[i].index = list(np.where(np.array(groupid_train)==i)) #index (順序非postid)
    if i not in global_group_test.keys(): continue
    group_test[i] = pd.DataFrame()
    group_test[i]['featurevec'] = list(np.array(global_group_test[i]))
    group_test[i].index = list(np.where(np.array(groupid_test)==i))

#%%
all_group_tr = pd.DataFrame()
for key, sub_df in group_train.items():
    all_group_tr = all_group_tr.append(sub_df, ignore_index=False) 
all_group_te = pd.DataFrame()
for key, sub_df in group_test.items():
    all_group_te = all_group_te.append(sub_df, ignore_index=False) 
all_group_tr = all_group_tr.sort_index()
all_group_te = all_group_te.sort_index()

group_tr_arr = np.array(all_group_tr.featurevec.tolist())
group_te_arr = np.array(all_group_te.featurevec.tolist())

#save arr
tr_arr_reshaped = group_tr_arr.reshape(group_tr_arr.shape[0], -1)
np.savetxt(datadir+"group_tr_arr.txt", tr_arr_reshaped)
te_arr_reshaped = group_te_arr.reshape(group_te_arr.shape[0], -1)
np.savetxt(datadir+"group_te_arr.txt", te_arr_reshaped)

#%%reload arr
loaded_arr = np.loadtxt(datadir+"group_tr_arr.txt")
group_tr_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"group_te_arr.txt")
group_te_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)

#%% each channel run one CNN model
import os
datadir = 'C:/Users/shuting/Desktop/論文/data/'
chanel_tr['postid'] = train_id
chanel_te['postid'] = test_id
chanel_tr['tag'] = list(tag_train)
chanel_te['tag'] = list(tag_test)
global_channel_train = {} #放CNN完後的feature, key是chanel number
global_channel_test = {}
if os.path.exists(datadir+"channelCNN_predict.txt"):
    os.remove(datadir+"channelCNN_predict.txt")
f = open(datadir+"channelCNN_predict.txt","w+")
for i in np.unique(chanel_tr.chanel_num):
    local_tr = chanel_tr[chanel_tr.chanel_num==i]
    local_tag_train = local_tr.tag
    local_tag_train = np.vstack(local_tag_train)
    embedding_channel_train = pd.DataFrame()
    embedding_channel_train['embedding'] = list((embedding_train))
    embedding_channel_train.index = train_id
    local_tr = embedding_channel_train['embedding'].loc[local_tr.postid]
    embedding_channel_train = np.zeros((len(local_tr),embedding_train.shape[1],embedding_train.shape[2]))
    for j in range(len(local_tr)):
        embedding_channel_train[j] = local_tr[j]

    local_te = chanel_te[chanel_te.chanel_num==i]
    if len(local_te)==0:
        print('---------------------------------', file=f)
        print('no test example in channel '+i, file=f)
        print('---------------------------------', file=f)
        continue
    local_tag_test = local_te.tag
    local_tag_test = np.vstack(local_tag_test)
    embedding_channel_test = pd.DataFrame()
    embedding_channel_test['embedding'] = list((embedding_test))
    embedding_channel_test.index = test_id
    local_te = embedding_channel_test['embedding'].loc[local_te.postid]
    embedding_channel_test = np.zeros((len(local_te),embedding_test.shape[1],embedding_test.shape[2]))
    for j in range(len(local_te)):
        embedding_channel_test[j] = local_te[j]

    gCNN = groupcnn()
    gCNN.fit(embedding_channel_train, local_tag_train, epochs=50, batch_size=4,
        validation_data=(embedding_channel_test, local_tag_test), callbacks=[early_stopping])
    gExtraction = Model(inputs=gCNN.input, outputs=gCNN.get_layer('groupfeature').output)
    y_pred = gCNN.predict(embedding_channel_test)
    acc_K, precision_K, recall_K, f1_K = evaluation(local_tag_test, y_pred, TopK)
    print('---------------------------------', file=f)
    print('channel: '+i, file=f)
    print('number of trainset: %d'%len(local_tr), file=f)
    print('number of testset: %d'%len(local_te), file=f)
    print('acc: %f'%acc_K, file=f)
    print('precision: %f'%precision_K, file=f)
    print('recall: %f'%recall_K, file=f)
    print('f1: %f'%f1_K, file=f)
    print('---------------------------------', file=f)
    global_channel_train[i] = gExtraction(embedding_channel_train)
    global_channel_test[i] = gExtraction(embedding_channel_test)
f.close()

#%%
channel_train = {}
channel_test = {} 
for i in global_channel_train:
    channel_train[i] = pd.DataFrame()
    channel_train[i]['featurevec'] = list(np.array(global_channel_train[i]))
    channel_train[i].index = chanel_tr[chanel_tr.chanel_num==i].postid
    channel_test[i] = pd.DataFrame()
    channel_test[i]['featurevec'] = list(np.array(global_channel_test[i]))
    channel_test[i].index = chanel_te[chanel_te.chanel_num==i].postid

#%%
channel_tr_arr = np.zeros((len(train_id), seq_length, embedding_size))
channel_te_arr = np.zeros((len(test_id), seq_length, embedding_size))
for i in range(len(train_id)):
    postid = train_id[i]
    key = chanel_tr.loc[i].chanel_num
    if key in channel_train.keys():
        channel_tr_arr[i] = channel_train[key].loc[postid].featurevec
 
for i in range(len(test_id)):
    postid = test_id[i]
    key = chanel_te.loc[i].chanel_num
    if key in channel_test.keys():
        channel_te_arr[i] = channel_test[key].loc[postid].featurevec

#%% save pretrain channel feature vector to database
'''
conn = MongoClient('localhost', 27017) #連結mongodb
db = conn.NiusNews2020_04_12 #create database
channel_train_db = db['channel_train_db'] #create text train collection  
channel_train_db.drop() #確定資料庫為空的
channel_test_db.insert(channel_train)
print('共有%s筆資料' % channel_train_db.count()) 

channel_test_db = db['channel_test_db'] #create text test collection  
channel_test_db.drop() #確定資料庫為空的
channel_test_db.insert(channel_test)
print('共有%s筆資料' % channel_test_db.count()) 
'''

#%% each lda topic run one CNN model
if os.path.exists(datadir+"ldaCNN_predict.txt"):
    os.remove(datadir+"ldaCNN_predict.txt")
f = open(datadir+"ldaCNN_predict.txt","w+")
global_lda_train = {} #放CNN完後的feature, key是chanel number
global_lda_test = {}
for i in range(n_clusters):
    local_trainid = np.where(np.array(doc_topic)==i)
    local_testid = np.where(np.array(doc_topic_te)==i)
    embedding_lda_train = embedding_train[local_trainid]
    embedding_lda_test = embedding_test[local_testid]

    if len(embedding_lda_test)==0:
        print('---------------------------------', file=f)
        print('no test example in topic '+str(i), file=f)
        print('---------------------------------', file=f)
        global_lda_train[i] = np.zeros((len(embedding_lda_train),seq_length,embedding_size))
        continue
    local_tag_train = tag_train[local_trainid]
    local_tag_test = tag_test[local_testid]

    gCNN = groupcnn()
    gCNN.fit(embedding_lda_train, local_tag_train, epochs=50, batch_size=4,
        validation_data=(embedding_lda_test, local_tag_test), callbacks=[early_stopping])
    gExtraction = Model(inputs=gCNN.input, outputs=gCNN.get_layer('groupfeature').output)
    y_pred = gCNN.predict(embedding_lda_test)
    acc_K, precision_K, recall_K, f1_K, ndcg_K = evaluation(local_tag_test, y_pred, TopK)
    print('---------------------------------', file=f)
    print('channel: %d'%i, file=f)
    print('number of trainset: %d'%len(embedding_lda_train), file=f)
    print('number of testset: %d'%len(embedding_lda_test), file=f)
    print('acc: %f'%acc_K, file=f)
    print('precision: %f'%precision_K, file=f)
    print('recall: %f'%recall_K, file=f)
    print('f1: %f'%f1_K, file=f)
    print('f1: %f'%ndcg_K, file=f)
    print('---------------------------------', file=f)
    global_lda_train[i] = gExtraction(embedding_lda_train)
    global_lda_test[i] = gExtraction(embedding_lda_test)
f.close()

#%%
lda_train = {}
lda_test = {} 
for i in global_lda_train:
    lda_train[i] = pd.DataFrame()
    lda_train[i]['featurevec'] = list(np.array(global_lda_train[i]))
    lda_train[i].index = list(np.where(np.array(doc_topic)==i)) #index (順序非postid)
    if i not in global_lda_test.keys(): continue
    lda_test[i] = pd.DataFrame()
    lda_test[i]['featurevec'] = list(np.array(global_lda_test[i]))
    lda_test[i].index = list(np.where(np.array(doc_topic_te)==i))

#%%
all_lda_tr = pd.DataFrame()
for key, sub_df in lda_train.items():
    all_lda_tr = all_lda_tr.append(sub_df, ignore_index=False) 
all_lda_te = pd.DataFrame()
for key, sub_df in lda_test.items():
    all_lda_te = all_lda_te.append(sub_df, ignore_index=False) 
all_lda_tr = all_lda_tr.sort_index()
all_lda_te = all_lda_te.sort_index()

lda_tr_arr = np.array(all_lda_tr.featurevec.tolist())
lda_te_arr = np.array(all_lda_te.featurevec.tolist())

#save arr
tr_arr_reshaped = lda_tr_arr.reshape(lda_tr_arr.shape[0], -1)
np.savetxt(datadir+"lda_tr_arr.txt", tr_arr_reshaped)
te_arr_reshaped = lda_te_arr.reshape(lda_te_arr.shape[0], -1)
np.savetxt(datadir+"lda_te_arr.txt", te_arr_reshaped)

#%%reload arr
loaded_arr = np.loadtxt(datadir+"lda_tr_arr.txt")
lda_tr_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)
loaded_arr = np.loadtxt(datadir+"lda_te_arr.txt")
lda_te_arr = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1]//embedding_size, embedding_size)

# %%
