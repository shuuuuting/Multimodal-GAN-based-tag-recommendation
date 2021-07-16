#%%----------------------------------------------------------------------------------------
#seldDef.py
#------------------------------------------------------------------------------------------
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
import keras
class coAttention_alt(Layer):
    """
    alternative co-attention
    inputs: [image feature tensor, hidden text feature tensor]
    output: co-Attention feature of image and text
    input dimensions:[(batchSize, num_region, CNN_dimension),
                    (batchSize, seq_length, CNN_dimension)]
    output dimension: batch_size*CNN_dimension
    """
    def __init__(self, dim_k, name='co_attn_layer'):
        super(coAttention_alt, self).__init__(name=name)
        self.dim_k = dim_k  # internal tensor dimension
        # self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def build(self, input_shape): #build:定義權重的function
        if not isinstance(input_shape, list): #input_shape要是list
            raise ValueError('A Co-Attention_alt layer should be called '
                             'on a list of inputs.')
        if len(input_shape) != 4: #input應該包含4個元件:img,text,group,title
            raise ValueError('A Co-Attention_alt layer should be called on a list of 4 inputs.'
                             'Got '+str(len(input_shape))+'inputs.')
        # print(input_shape)
        self.num_imgRegion = input_shape[0][1]
        self.seq_len = input_shape[1][1]
        self.group_len = input_shape[2][1]
        self.output_dim = input_shape[0][2] #200
        self.title_len = input_shape[3][1] #10

        self.w_Dense_Vi_0 = self.add_weight(name='w_Dense_Vi_0',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Vt_0 = self.add_weight(name='w_Dense_Vt_0',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Pi_0 = self.add_weight(name='w_Dense_Pi_0',
                                            shape=(2*self.dim_k, 1),
                                            initializer='random_normal',
                                            trainable=True)
        self.b_Dense_Pi_0 = self.add_weight(name='b_Dense_Pi_0',
                                            shape=(self.num_imgRegion,),
                                            initializer='zeros',
                                            trainable=True)
        self.w_Dense_Vi_1 = self.add_weight(name='w_Dense_Vi_1',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Vt_1 = self.add_weight(name='w_Dense_Vt_1',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Pi_1 = self.add_weight(name='w_Dense_Pi_1',
                                            shape=(2*self.dim_k, 1),
                                            initializer='random_normal',
                                            trainable=True)
        self.b_Dense_Pi_1 = self.add_weight(name='b_Dense_Pi_1',
                                            shape=(self.seq_len,),
                                            initializer='zeros',
                                            trainable=True)                    
        self.w_Dense_Vi_2 = self.add_weight(name='w_Dense_Vi_2',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Vt_2 = self.add_weight(name='w_Dense_Vt_2',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Pi_2 = self.add_weight(name='w_Dense_Pi_2',
                                            shape=(2*self.dim_k, 1),
                                            initializer='random_normal',
                                            trainable=True)
        self.b_Dense_Pi_2 = self.add_weight(name='b_Dense_Pi_2',
                                            shape=(self.group_len,), #[2-1]self.seq_len,[2-2]self.title_len
                                            initializer='zeros',
                                            trainable=True)
        self.w_Dense_Vi_3 = self.add_weight(name='w_Dense_Vi_3',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Vt_3 = self.add_weight(name='w_Dense_Vt_3',
                                            shape=(self.output_dim, self.dim_k),
                                            initializer='random_normal',
                                            trainable=True)
        self.w_Dense_Pi_3 = self.add_weight(name='w_Dense_Pi_3',
                                            shape=(2*self.dim_k, 1),
                                            initializer='random_normal',
                                            trainable=True)
        self.b_Dense_Pi_3 = self.add_weight(name='b_Dense_Pi_3',
                                            shape=(self.title_len,), #[2-1]self.seq_len,[2-2]self.title_len
                                            initializer='zeros',
                                            trainable=True)

        super(coAttention_alt, self).build(input_shape)  # Be sure to call this somewhere! 一定要在最後調用它!

    def call(self, x, mask=None): #call:編寫layer邏輯的function，執行forward propagation
        ifeature = x[0]
        tfeature_h = x[1]
        # tfeature = x[2]
        gfeature = x[2]
        efeature = x[3]
        output_dim = self.output_dim
        num_imgRegion = self.num_imgRegion
        dim_k = self.dim_k
        seq_len = self.seq_len
        tfeature = K.mean(tfeature_h, axis=1)

        # phase 0: text-guided image feature computation
        w_Vi_0 = K.dot(K.reshape(ifeature, [-1, output_dim]), self.w_Dense_Vi_0) # shape=((batchSize*num_imgRegion),dim_k)
        w_Vi_0 = K.reshape(w_Vi_0, [-1, num_imgRegion, dim_k])  # shape=(batchSize,num_imgRegion,dim_k)
        w_Vt_0 = K.repeat(K.dot(tfeature, self.w_Dense_Vt_0), num_imgRegion)  # shape=(batchSize,num_imgRegion,dim_k) #未repeat前的dim是(batchSize,dim_k)
        Vi_Vt_0 = K.concatenate([w_Vi_0, w_Vt_0], axis=-1)  # shape=(batchSize,num_imgRegion,2*dim_k)
        Hi = K.tanh(Vi_Vt_0)
        # Hi_w = K.squeeze(K.dot(K.reshape(Hi, [-1, 2*dim_k]), self.w_Dense_Pi_0), axis=-1)
        # Hi_w_b = K.reshape(Hi_w, [-1, num_imgRegion]) + self.b_Dense_Pi_0
        Hi_w_b = K.squeeze(K.dot(Hi, self.w_Dense_Pi_0), axis=-1) + self.b_Dense_Pi_0  # shape=(batchSize,num_imgRegion) #axis是要丟棄的軸 #squeeze是用來刪掉維數是1的維度
        Pi = K.softmax(Hi_w_b) # shape=(batchSize,num_imgRegion)
        Pi = K.permute_dimensions(K.repeat(Pi, output_dim), (0, 2, 1))  # shape=(batchSize,num_imgRegion,output_dim)
        Pi_Vi = Pi*ifeature # shape=(batchSize,num_imgRegion,output_dim)*(batchSize,num_imgRegion,output_dim)
        Vi = K.sum(Pi_Vi, axis=1)  # shape=(batchSize,output_dim)

        # phase 1: image-guided text feature computation
        w_Vi_1 = K.repeat(K.dot(Vi, self.w_Dense_Vi_1), seq_len)    # shape=(batchSize,seq_len,dim_k)
        w_Vt_1 = K.dot(K.reshape(tfeature_h, [-1, output_dim]), self.w_Dense_Vt_1)   # shape=((batchSize*seq_len),dim_k)
        w_Vt_1 = K.reshape(w_Vt_1, (-1, seq_len, dim_k))    # shape= (batchSize,seq_len,dim_k)
        Vi_Vt_1 = K.concatenate([w_Vi_1, w_Vt_1], axis=-1)    # shape=(batchSize,seq_len,2*dim_k)
        Ht = K.tanh(Vi_Vt_1)
        Ht_b = K.squeeze(K.dot(Ht, self.w_Dense_Pi_1), axis=-1) + self.b_Dense_Pi_1   # shape=(batch_size,seq_len)
        Pt = K.softmax(Ht_b)
        Pt = K.permute_dimensions(K.repeat(Pt, output_dim), (0, 2, 1))    # shape=(batchSize,seq_len,output_dim)
        Pt_Vt = Pt*tfeature_h # shape=(batchSize,seq_len,output_dim)*(batchSize,seq_len,output_dim)
        Vt = K.sum(Pt_Vt, axis=1)   # shape=(batchSize,output_dim)
        
        # phase 2: co-guided popular feature 
        w_Vi_2 = K.repeat(K.dot(Vi+Vt, self.w_Dense_Vi_2), self.group_len)    # shape=(batchSize,seq_len,dim_k)
        w_Vt_2 = K.dot(K.reshape(gfeature, [-1, output_dim]), self.w_Dense_Vt_2)   # shape=((batchSize*seq_len),dim_k)
        w_Vt_2 = K.reshape(w_Vt_2, (-1, self.group_len, dim_k))    # shape= (batchSize,seq_len,dim_k)
        Vi_Vt_2 = K.concatenate([w_Vi_2, w_Vt_2], axis=-1)    # shape=(batchSize,seq_len,2*dim_k)
        Hg = K.tanh(Vi_Vt_2)
        Hg_b = K.squeeze(K.dot(Hg, self.w_Dense_Pi_2), axis=-1) + self.b_Dense_Pi_2   # shape=(batch_size,seq_len)
        Pg = K.softmax(Hg_b)
        Pg = K.permute_dimensions(K.repeat(Pg, output_dim), (0, 2, 1))    # shape=(batchSize,seq_len,output_dim)
        Pg_Vt = Pg*gfeature # shape=(batchSize,seq_len,output_dim)*(batchSize,seq_len,output_dim)
        Vg = K.sum(Pg_Vt, axis=1)   # shape=(batchSize,output_dim)
        
        # phase 3: co-guided multi-gCNN feature
        w_Vi_3 = K.repeat(K.dot(Vi+Vt, self.w_Dense_Vi_3), self.title_len)    # shape=(batchSize,10,dim_k)
        w_Vt_3 = K.dot(K.reshape(efeature, [-1, output_dim]), self.w_Dense_Vt_3)   # shape=((batchSize*10),dim_k)
        w_Vt_3 = K.reshape(w_Vt_3, (-1, self.title_len, dim_k))    # shape= (batchSize,10,dim_k)
        Vi_Vt_3 = K.concatenate([w_Vi_3, w_Vt_3], axis=-1)    # shape=(batchSize,10,2*dim_k)
        He = K.tanh(Vi_Vt_3)
        He_b = K.squeeze(K.dot(He, self.w_Dense_Pi_3), axis=-1) + self.b_Dense_Pi_3   # shape=(batch_size,10)
        Pe = K.softmax(He_b)
        Pe = K.permute_dimensions(K.repeat(Pe, output_dim), (0, 2, 1))    # shape=(batchSize,10,output_dim)
        Pe_Vt = Pe*efeature # shape=(batchSize,10,output_dim)*(batchSize,10,output_dim)
        Ve = K.sum(Pe_Vt, axis=1)   # shape=(batchSize,output_dim)
        alpha = 0.5
        beta = 0.5
        return Vi + Vt + alpha*Vg + beta*Ve
        #return K.concatenate([Vi,Vt,Vg,Ve], axis=-1)

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][-1]) #(batch_size,CNN_dimension)
        return output_shape

    def get_config(self):
        return super(coAttention_alt, self).get_config()

def myLossFunc(y_true, y_pred):
    probs_log = -K.log(y_pred)
    loss = K.mean(K.sum(probs_log*y_true, axis=-1))
    # loss = K.mean(K.sum(K.clip(probs_log * y_true, -1e40, 100), axis=-1))
    return loss

#%%----------------------------------------------------------------------------------------
#co-attention.py 
#------------------------------------------------------------------------------------------
from keras.models import Model
from keras.layers.core import Activation, Flatten, Reshape, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import AveragePooling1D,Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Dense, Embedding, Dropout, Lambda

num_tags = tag_train.shape[1]
num_words = vocab_size
#index_from = 3
seq_length = max_length
title_length = 10
batch_size = 64
embedding_size = 200
hidden_size = 100
attention_size = 200
dim_k = 100
num_region = 7*7
drop_rate = 0.5
TopK = 10

def imageFeature(inputs):
    features = Reshape(target_shape=(num_region, 512))(inputs)
    features = Dense(embedding_size, activation="tanh", use_bias=False)(features) #single layer to convert each img vector into a new same dim vector as text feature vector
    features = SeqSelfAttention(attention_activation='sigmoid')(features)
    features_pooling = AveragePooling1D(pool_size=num_region, padding="same")(features)
    features_pooling = Lambda(lambda x: K.squeeze(x, axis=1))(features_pooling)

    return features, features_pooling

def textFeature(X):
    #embeddings = Embedding(input_dim=num_words, output_dim=embedding_size,
    #                       mask_zero=True, input_length=seq_length)(X) #seq_length:代表取前幾個熱門字
    #tFeature = LSTM(units=embedding_size, return_sequences=True)(embeddings)
    #tFeature = Bidirectional(LSTM(units=embedding_size, return_sequences=True), merge_mode='sum')(embeddings)
    tFeature1 = Conv1D(embedding_size, 3, padding='same', strides=1, activation='relu')(X)
    tFeature2 = Conv1D(embedding_size, 4, padding='same', strides=1, activation='relu')(X)
    tFeature3 = Conv1D(embedding_size, 5, padding='same', strides=1, activation='relu')(X)
    tFeature = Add(name='tcnn')([tFeature1, tFeature2, tFeature3])
    tFeature = SeqSelfAttention(attention_activation='sigmoid')(tFeature)

    return tFeature

def groupFeature(X):
    #X = Embedding(input_dim=num_words, output_dim=embedding_size,
    #                       mask_zero=True, input_length=seq_length)(X) #seq_length:一次輸入带有的詞彙個數
    #gFeature = LSTM(units=embedding_size, return_sequences=True)(embeddings)
    #gFeature = Bidirectional(LSTM(units=embedding_size, return_sequences=True), merge_mode='sum')(embeddings)
    gFeature1 = Conv1D(embedding_size, 3, padding='same', strides=1, activation='relu')(X)
    gFeature2 = Conv1D(embedding_size, 4, padding='same', strides=1, activation='relu')(X)
    gFeature3 = Conv1D(embedding_size, 5, padding='same', strides=1, activation='relu')(X)
    gFeature = Add()([gFeature1, gFeature2, gFeature3])
    gFeature = SeqSelfAttention(attention_activation='sigmoid')(gFeature)
    #gFeature = K.mean(gFeature, axis=1) #用phase2-2要註解掉這條
    return gFeature

def titleFeature(X):
    eFeature1 = Conv1D(embedding_size, 3, padding='same', strides=1, activation='relu')(X)
    eFeature2 = Conv1D(embedding_size, 4, padding='same', strides=1, activation='relu')(X)
    eFeature3 = Conv1D(embedding_size, 5, padding='same', strides=1, activation='relu')(X)
    eFeature = Add()([eFeature1, eFeature2, eFeature3])
    eFeature = SeqSelfAttention(attention_activation='sigmoid')(eFeature)
    return eFeature

def modelDef():
    inputs_img = Input(shape=(7, 7, 512))
    #inputs_text = Input(shape=(seq_length,embedding_size))
    inputs_text = Input(shape=(seq_length+title_length,embedding_size)) #concate w/ title
    inputs_channel = Input(shape=(seq_length,embedding_size)) #embedding
    inputs_title = Input(shape=(seq_length+title_length,embedding_size)) #embedding 

    iFeature, iFeature_pooling = imageFeature(inputs_img)
    tFeature = textFeature(inputs_text)
    '''
    gCNN = load_model('pretrained_gCNN.h5', custom_objects={"SeqSelfAttention": SeqSelfAttention})
    gCNN = Model(inputs=gCNN.input, outputs=gCNN.get_layer('gcnn').output)
    gFeature = gCNN(inputs_channel)
    gFeature = SeqSelfAttention(attention_activation='sigmoid')(gFeature)
    '''
    gFeature = groupFeature(inputs_channel) #embedding
    gFeature = SeqSelfAttention(attention_activation='sigmoid')(inputs_channel)
    eFeature = groupFeature(inputs_title) #embedding
    #eFeature = SeqSelfAttention(attention_activation='sigmoid')(inputs_title)

    iFeature.set_shape((inputs_img.shape[0],num_region,embedding_size))
    #tFeature.set_shape((inputs_text.shape[0],seq_length,embedding_size))
    tFeature.set_shape((inputs_text.shape[0],seq_length+title_length,embedding_size)) #concate w/ title
    gFeature.set_shape((inputs_channel.shape[0],seq_length,embedding_size))
    eFeature.set_shape((inputs_title.shape[0],seq_length+title_length,embedding_size))

    co_feature = coAttention_alt(dim_k=dim_k)([iFeature, tFeature, gFeature, eFeature])
    dropout = Dropout(drop_rate)(co_feature)
    #all_feature = attn(dropout, gFeature)
    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
    Sigmoid = Dense(num_tags, activation="sigmoid", use_bias=True)(dropout)
    model = Model(inputs=[inputs_img, inputs_text, inputs_channel, inputs_title],
                  outputs=[Softmax])
    # adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    #model.compile(optimizer="adam", loss='BinaryCrossentropy')
    model.compile(optimizer="adam", loss=myLossFunc)
    return model

def evaluation(y_true, y_pred, top_K):
    acc_count = 0
    precision_K = []
    recall_K = []
    f1_K = []

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:] #每篇文章排名前10可能的tag index
        if np.sum(y_true[i, top_indices]) >= 1: #代表預測出來要推薦的tag有hit到真實tag
            acc_count += 1
        p = np.sum(y_true[i, top_indices])/top_K
        r = np.sum(y_true[i, top_indices])/np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
        if p != 0 or r != 0:
            f1_K.append(2 * p * r / (p + r))
        else:
            f1_K.append(0)
    acc_K = acc_count * 1.0 / y_pred.shape[0]

    return acc_K, np.mean(np.array(precision_K)), np.mean(np.array(recall_K)), np.mean(np.array(f1_K))

#%%----------------------------------------------------------------------------------------
#use kmeans to group popular words (w2v)
#------------------------------------------------------------------------------------------
popularwords = [' '.join(map(str, l)) for l in popularcontent] 
group_data = []
i = 0
for d in popularwords:
    group_data.append([])
    for w in d:
        try:
            group_data[i].append(w2vmodel.wv[w])
        except:
            group_data[i].append(np.zeros(200))
    i+=1
#group_data.append(np.mean(tmp, axis=0))
group_data = pad_sequences(group_data, dtype='float32', maxlen=max_length, padding='post')
group_train = np.array([group_data[i] for i in groupid_train]) #(,50,200)
group_test = np.array([group_data[i] for i in groupid_test])

#%%----------------------------------------------------------------------------------------
#use lda to group popular words (w2v)
#------------------------------------------------------------------------------------------
popularwords = [' '.join(map(str, l)) for l in ldapopularcontent] 
group_data = []
i = 0
for d in popularwords:
    group_data.append([])
    for w in d:
        try:
            group_data[i].append(w2vmodel.wv[w])
        except:
            group_data[i].append(np.zeros(200))
    i+=1
#group_data.append(np.mean(tmp, axis=0))
group_data = pad_sequences(group_data, dtype='float32', maxlen=max_length, padding='post')
lda_train = np.array([group_data[i] for i in doc_topic]) #(,50,200)
lda_test = np.array([group_data[i] for i in doc_topic_te])

#%%----------------------------------------------------------------------------------------
#加入title資料做attention
#------------------------------------------------------------------------------------------
title = pd.DataFrame(list(postsdb.find({},{"_id": 0,"postid": 1,"title": 1})))
title = title.set_index('postid')
title_train = title.loc[list(text_tr_pd.postid)]
title_test = title.loc[list(text_te_pd.postid)]
title_train['sentence'] = [' '.join(map(str, l)) for l in title_train['title']]  
title_test['sentence'] = [' '.join(map(str, l)) for l in title_test['title']] 
title_train = [one_hot(d, vocab_size) for d in list(title_train['sentence'])]
title_train = pad_sequences(title_train, maxlen=10, padding='post') #(,10)
title_test = [one_hot(d, vocab_size) for d in list(title_test['sentence'])]
title_test = pad_sequences(title_test, maxlen=10, padding='post')
#要用這個跑的話，seq_len要同樣改為10
#seq_length = 10

#%%----------------------------------------------------------------------------------------
#run co-attention.py 
#------------------------------------------------------------------------------------------
from keras.layers import Attention
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True) #持續5個epoch沒下降就停
model = modelDef()
text_title_train = np.concatenate((embedding_train,embedding_title_train),axis=1)
text_title_test = np.concatenate((embedding_test,embedding_title_test),axis=1)
#text_title_train = Attention()([embedding_train,embedding_title_train])
#text_title_test = Attention()([embedding_test,embedding_title_test])
history = model.fit(x=[image_train, text_title_train, group_train, group_title_tr_arr], #group_tr_arr  lda_tr_arr group_train lda_train
                    y=tag_train.astype(np.float32),
                    batch_size=batch_size,
                    epochs=50,
                    verbose=1,)
                    #validation_data=([image_test, embedding_test, channel_te_arr, embedding_title_test], tag_test.astype(np.float32)), 
                    #callbacks=[early_stopping])
y_pred = model.predict(x=[image_test, text_title_test, group_test, group_title_te_arr]) #group_te_arr  lda_te_arr group_test lda_test
acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)

#%% phase 2 i+t guide channel
# + phase 3 i+t guide title 
#alpha=0.5/beta=0.5
#acc:  0.8298059964726632
#precision:  0.16155202821869488
#recall:  0.7228720500545897
#f1:  0.2495514166140896

#%% pretrain word2vec+(i&t)selfattention / no pretrain model
#alpha=0.5/beta=0.5
#acc:  0.845679012345679
#precision:  0.16763668430335096
#recall:  0.7483430615044372
#f1:  0.25898622844828223

#%% pretrain word2vec+(i&t&c&e)selfattention / no pretrain model
#alpha=0.5/beta=0.5
#acc:  0.855379188712522
#precision:  0.17116402116402119
#recall:  0.7639417289549565
#f1:  0.2643715746892398