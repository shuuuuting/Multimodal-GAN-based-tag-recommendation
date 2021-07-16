#以個別文章資料當輸入、群組的tag機率當作正確答案去訓練單一個模型，用於做pretrained gCNN
#%%
'''
tags.index = tags.postid
tag_tr_pd = tags.loc[train_id]
#tag_te_pd = tags.loc[test_id]
tag_tr_pd = tag_tr_pd.reset_index(drop=True)
'''

#%%compute group tag probability as ground truth
g = globals()
group_tag = []
for i in range(n_clusters):
    varname = 'c{}'.format(i)
    local_tag = g[varname].tag
    local_tag = np.vstack(local_tag)
    local_tag = np.mean(local_tag, axis=0)
    group_tag.append(local_tag)
groupid_train = cluster_map['cluster']
group_tag_train = np.array([group_tag[i] for i in groupid_train]) 
groupid_test = kmeans_model.predict(np.array([d2v_model.infer_vector(i, steps=50, alpha=0.025) for i in text_te_pd.content_jieba]))
group_tag_test = np.array([group_tag[i] for i in groupid_test])

#%%compute channel tag probability as ground truth
chanel_tr['tag'] = list(tag_train)
channel_tag = {}
for i in np.unique(chanel_tr.chanel_num):
    local = chanel_tr[chanel_tr.chanel_num==i]
    local_tag = local.tag
    local_tag = np.vstack(local_tag)
    local_tag = np.mean(local_tag, axis=0)
    channel_tag[i] = local_tag
channel_tag_train = np.array([channel_tag[i] for i in chanel_tr.chanel_num]) 
channel_tag_test = np.array([channel_tag[i] for i in chanel_te.chanel_num])

#%%compute lda topic tag probability as ground truth
g = globals()
lda_tag = []
for i in range(n_clusters):
    varname = 't{}'.format(i)
    local_tag = g[varname].tag
    local_tag = np.vstack(local_tag)
    local_tag = np.mean(local_tag, axis=0)
    lda_tag.append(local_tag)
ldaid_train = doc_topic
lda_tag_train = np.array([lda_tag[i] for i in ldaid_train]) 
ldaid_test = doc_topic_te
lda_tag_test = np.array([lda_tag[i] for i in ldaid_test])


#%%
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import MaxPooling1D,Conv1D
from keras.layers import Input, Dense, Embedding, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam

embedding_size = 200 
seq_length = 50
TopK = 5
batch_size = 64
optimizer = Adam(lr=0.0005, decay=0.000001)
num_tags = tag_train.shape[1]
def groupcnn():
    inp = Input(shape=(seq_length,embedding_size))
    conv1 = Conv1D(embedding_size, 3, padding='same', activation='relu')(inp)
    #conv1 = SeqSelfAttention(attention_activation='sigmoid')(conv1)
    #conv1 = MaxPooling1D()(conv1)
    
    conv2 = Conv1D(embedding_size, 4, padding='same', activation='relu')(inp)
    #conv2 = SeqSelfAttention(attention_activation='sigmoid')(conv2)
    #conv2 = MaxPooling1D()(conv2)
    
    conv3 = Conv1D(embedding_size, 5, padding='same', activation='relu')(inp)
    #conv3 = SeqSelfAttention(attention_activation='sigmoid')(conv3)
    #conv3 = MaxPooling1D()(conv3)
    
    #cnn = concatenate([conv1, conv2, conv3], axis=-1)
    cnn = Add(name='gcnn')([conv1, conv2, conv3])
    cnn = SeqSelfAttention(attention_activation='sigmoid')(cnn)
    cnn.set_shape((cnn.shape[0],seq_length,embedding_size))
    flat = Flatten()(cnn)
    #flat = GlobalAveragePooling1D()(cnn)
    #x = Dense(1000, activation='relu')(flat)
    #x = Dropout(0.5)(flat)
    #x = Dense(500, activation='relu')(x)
    x = Dropout(0.2)(flat)
    x = Dense(num_tags, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
model = groupcnn()

#%%
method = 'groupCNN'
wb = load_workbook(resultdir+'evaluation.xlsx')
ws = wb.create_sheet(title=method)
ws.append(['acc', 'precision', 'recall', 'f1', 'ndcg', 'map'])
if not os.path.exists(resultdir+method):
    os.makedirs(resultdir+method)

round = 10
for i in range(round):
    print('round', i+1)
    model.fit(lda_tr_arr, tag_train.astype(np.float32), epochs=50, batch_size=batch_size,)
    y_pred = model.predict(lda_te_arr)
    acc_K, precision_K, recall_K, f1_K, ndcg_K, map_K = evaluation(tag_test, y_pred, TopK)
    print('acc: ', acc_K)
    print('precision: ', precision_K)
    print('recall: ', recall_K)
    print('f1: ', f1_K)
    print('ndcg: ', ndcg_K)
    print('map: ', map_K)
    np.savetxt(resultdir+method+'/y_pred'+str(i+1)+'.csv', y_pred, delimiter=",")
    save_result(acc_K, precision_K, recall_K, f1_K, ndcg_K, map_K)
wb.save(filename = resultdir+'evaluation.xlsx')

#%% train by group tag prob
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True) #持續5個epoch沒下降就停
model.fit(embedding_train, lda_tag_train, epochs=100, batch_size=batch_size, #lda_tag_train group_tag_train
            validation_data=(embedding_test, lda_tag_test), callbacks=[early_stopping]) #lda_tag_test group_tag_test
model.save('pretrained_gCNN.h5')
y_pred = model.predict(embedding_test)
acc_K, precision_K, recall_K, f1_K = evaluation(lda_tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)


#%% group tag probability as ground truth
#acc:  0.003527336860670194
#precision:  0.04006781795303557
#recall:  0.19423388518471316
#f1:  0.06641323175769227
#reason: test是用原本kmeans model預測出來的，但我發現大部分預測出來的都在同一類

#%% group tag probability as ground truth(init by channel)
#acc:  0.0
#precision:  0.048855394440310874
#recall:  0.21974726631261007
#f1:  0.0799235868832429

#%% channel tag probability as ground truth
#acc:  0.4382716049382716
#precision:  0.08619409288270043
#recall:  0.3799116307838604
#f1:  0.1390010044192231

#%% lda tag probability as ground truth
#acc:  0.1111111111111111
#precision:  0.06573610949036884
#recall:  0.29220225239537934
#f1:  0.10658893598800831