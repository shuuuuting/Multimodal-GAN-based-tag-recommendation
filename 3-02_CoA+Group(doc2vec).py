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
        if len(input_shape) != 3: #input應該包含三個元件:img,text,group
            raise ValueError('A Co-Attention_alt layer should be called on a list of 3 inputs.'
                             'Got '+str(len(input_shape))+'inputs.')
        # print(input_shape)
        self.num_imgRegion = input_shape[0][1]
        self.seq_len = input_shape[1][1]
        self.output_dim = input_shape[0][2] #200

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
        self.w_co = self.add_weight(name='w_co',
                                    shape=(self.output_dim, self.dim_k),
                                    initializer='random_normal',
                                    trainable=True)
        self.w_group = self.add_weight(name='w_group',
                                    shape=(self.output_dim, self.output_dim),
                                    initializer='random_normal',
                                    trainable=True)                      
        self.alpha = self.add_weight(name='alpha', shape=(1,), 
                                    initializer=keras.initializers.Constant(value=0.8),
                                    trainable=True,
                                    constraint=keras.constraints.min_max_norm(max_value=1,min_value=0))
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
                                            shape=(self.seq_len,), #2-1:self.seq_len #2-2:10
                                            initializer='zeros',
                                            trainable=True)

        super(coAttention_alt, self).build(input_shape)  # Be sure to call this somewhere! 一定要在最後調用它!

    def call(self, x, mask=None): #call:編寫layer邏輯的function，執行forward propagation
        ifeature = x[0]
        tfeature_h = x[1]
        # tfeature = x[2]
        gfeature = x[2]
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
        #return Vi+Vt

        
        # phase 2-1: title-guided text feature 
        w_Vi_2 = K.repeat(K.dot(gfeature, self.w_Dense_Vi_2), seq_len)    # shape=(batchSize,seq_len,dim_k)
        w_Vt_2 = K.dot(K.reshape(tfeature_h, [-1, output_dim]), self.w_Dense_Vt_2)   # shape=((batchSize*seq_len),dim_k)
        w_Vt_2 = K.reshape(w_Vt_2, (-1, seq_len, dim_k))    # shape= (batchSize,seq_len,dim_k)
        Vi_Vt_2 = K.concatenate([w_Vi_2, w_Vt_2], axis=-1)    # shape=(batchSize,seq_len,2*dim_k)
        Hg = K.tanh(Vi_Vt_2)
        Hg_b = K.squeeze(K.dot(Hg, self.w_Dense_Pi_2), axis=-1) + self.b_Dense_Pi_2   # shape=(batch_size,seq_len)
        Pg = K.softmax(Hg_b)
        Pg = K.permute_dimensions(K.repeat(Pg, output_dim), (0, 2, 1))    # shape=(batchSize,seq_len,output_dim)
        Pg_Vt = Pg*tfeature_h # shape=(batchSize,seq_len,output_dim)*(batchSize,seq_len,output_dim)
        Vg = K.sum(Pg_Vt, axis=1)   # shape=(batchSize,output_dim)
        '''
        # phase 2-2: co-guided title feature
        w_Vi_2 = K.repeat(K.dot(Vi+Vt, self.w_Dense_Vi_2), 10)    # shape=(batchSize,10,dim_k)
        w_Vt_2 = K.dot(K.reshape(gfeature, [-1, output_dim]), self.w_Dense_Vt_2)   # shape=((batchSize*10),dim_k)
        w_Vt_2 = K.reshape(w_Vt_2, (-1, 10, dim_k))    # shape= (batchSize,10,dim_k)
        Vi_Vt_2 = K.concatenate([w_Vi_2, w_Vt_2], axis=-1)    # shape=(batchSize,10,2*dim_k)
        Hg = K.tanh(Vi_Vt_2)
        Hg_b = K.squeeze(K.dot(Hg, self.w_Dense_Pi_2), axis=-1) + self.b_Dense_Pi_2   # shape=(batch_size,10)
        Pg = K.softmax(Hg_b)
        Pg = K.permute_dimensions(K.repeat(Pg, output_dim), (0, 2, 1))    # shape=(batchSize,10,output_dim)
        Pg_Vt = Pg*gfeature # shape=(batchSize,10,output_dim)*(batchSize,10,output_dim)
        Vg = K.sum(Pg_Vt, axis=1)   # shape=(batchSize,output_dim)
        '''
        return Vi+Vt+0.5*Vg
        #merge co-feature and group feature
        '''
        co_feature = Vi+Vt # shape=(batchSize,output_dim)
        weighted_co = K.dot(co_feature, self.w_co) # shape=(batchSize,dim_k)
        weighted_group = K.dot(gfeature, self.w_group) # shape=(batchSize,dim_k)
        all_feature = K.concatenate([weighted_co, weighted_group], axis=-1) # shape=(batchSize,output_dim)
        #all_attn = K.softmax(K.tanh(all_feature))
        #K.print_tensor(self.alpha, message='alpha = ')
        #return self.alpha*co_feature + (1-self.alpha)*gfeature
        #return co_feature * gfeature
        #return co_feature + gfeature
        return co_feature * K.tanh(K.dot(gfeature, self.w_group))
        '''

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

#%%
def attn(co_feature, gFeature):
    concate = K.concatenate([co_feature, gFeature], axis=-1)
    attnweight = Dense(2, activation="softmax", use_bias=True)(concate)
    #return attnweight[:,0]*co_feature + attnweight[:,1]*gFeature
    return co_feature + gFeature

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
batch_size = 256
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
     #                      mask_zero=True, input_length=seq_length)(X) #seq_length:代表取前幾個熱門字
    #tFeature = LSTM(units=embedding_size, return_sequences=True)(embeddings)
    #tFeature = Bidirectional(LSTM(units=embedding_size, return_sequences=True), merge_mode='sum')(embeddings)
    tFeature1 = Conv1D(embedding_size, 3, padding='same', strides=1, activation='relu')(X)
    tFeature2 = Conv1D(embedding_size, 4, padding='same', strides=1, activation='relu')(X)
    tFeature3 = Conv1D(embedding_size, 5, padding='same', strides=1, activation='relu')(X)
    tFeature = Add(name='tcnn')([tFeature1, tFeature2, tFeature3])
    tFeature = SeqSelfAttention(attention_activation='sigmoid')(tFeature)

    return tFeature

def groupFeature(X):
    X = Embedding(input_dim=num_words, output_dim=embedding_size,
                           mask_zero=True, input_length=seq_length)(X) #seq_length:一次輸入带有的詞彙個數
    #gFeature = LSTM(units=embedding_size, return_sequences=True)(embeddings)
    #gFeature = Bidirectional(LSTM(units=embedding_size, return_sequences=True), merge_mode='sum')(embeddings)
    gFeature1 = Conv1D(embedding_size, 3, padding='same', strides=1, activation='relu')(X)
    gFeature2 = Conv1D(embedding_size, 4, padding='same', strides=1, activation='relu')(X)
    gFeature3 = Conv1D(embedding_size, 5, padding='same', strides=1, activation='relu')(X)
    gFeature = Add()([gFeature1, gFeature2, gFeature3])
    gFeature = SeqSelfAttention(attention_activation='sigmoid')(gFeature)
    #gFeature = K.mean(gFeature, axis=1)
    return gFeature

def modelDef():
    inputs_img = Input(shape=(7, 7, 512))
    inputs_text = Input(shape=(seq_length, embedding_size))
    #inputs_group = Input(shape=(seq_length,)) #embedding
    inputs_group = Input(shape=(embedding_size,)) #doc2vec
    iFeature, iFeature_pooling = imageFeature(inputs_img)
    tFeature = textFeature(inputs_text)
    iFeature.set_shape((inputs_img.shape[0], num_region, embedding_size))
    tFeature.set_shape((inputs_text.shape[0], seq_length, embedding_size))
    #gFeature = groupFeature(inputs_group) #embedding
    gFeature = inputs_group #doc2vec
    co_feature = coAttention_alt(dim_k=dim_k)([iFeature, tFeature, gFeature])
    dropout = Dropout(drop_rate)(co_feature)
    #all_feature = attn(dropout, gFeature)
    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
    Sigmoid = Dense(num_tags, activation="sigmoid", use_bias=True)(dropout)
    model = Model(inputs=[inputs_img, inputs_text, inputs_group],
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
#popularcontent as group feature (doc2vec)
#------------------------------------------------------------------------------------------
group_vec = [d2v_model.infer_vector(i, steps=50, alpha=0.025) for i in popularcontent]
groupid_train = X.labels_
group_train = np.array([group_vec[i] for i in groupid_train]) #(,200)
groupid_test = kmeans_model.predict(np.array([d2v_model.infer_vector(i, steps=50, alpha=0.025) for i in text_te_pd.content_jieba]))
group_test = np.array([group_vec[i] for i in groupid_test])

#%%----------------------------------------------------------------------------------------
#use lda to group popular words (doc2vec)
#------------------------------------------------------------------------------------------
group_data = [d2v_model.infer_vector(d, steps=50, alpha=0.025) for d in ldapopularcontent]
group_train = np.array([group_data[i] for i in doc_topic]) #(,50)
group_test = np.array([group_data[i] for i in doc_topic_te])

#%%----------------------------------------------------------------------------------------
#run co-attention.py 
#------------------------------------------------------------------------------------------
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True) #持續5個epoch沒下降就停
model = modelDef()
history = model.fit(x=[image_train, embedding_train, group_train], #group_train
                    y=tag_train.astype(np.float32),
                    batch_size=batch_size,
                    epochs=50,
                    verbose=1,)
                    #validation_data=([image_test, embedding_test, group_test], tag_test.astype(np.float32)), 
                    #callbacks=[early_stopping])
y_pred = model.predict(x=[image_test, embedding_test, group_test]) #group_test
acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)

#%% Kmeans + Doc2Vec
# [ 單純相加 co_feature + gFeature ]
#acc:  0.6472898664571878
#precision:  0.1267085624509034
#recall:  0.496097519919201
#f1:  0.1905927277185537

# [ 用alpha調整兩者權重 alpha init=1 ]
#acc:  0.648075412411626
#precision:  0.12757266300078557
#recall:  0.5005134103916508
#f1:  0.1920424097874308

# [ 用alpha調整兩者權重 alpha init=0.5 ]
#acc:  0.6001571091908877
#precision:  0.11822466614296936
#recall:  0.4528691130812105
#f1:  0.1770663789332058

# [ phase2-1 (CNN) ]
# -- 451 tags --
#acc:  0.685243328100471
#precision:  0.1402668759811617
#recall:  0.543711220752037
#f1:  0.20965407641578107
# -- 263 tags --
#acc:  0.7530864197530864
#precision:  0.14832451499118166
#recall:  0.6458532795834383
#f1:  0.2281228122217442

#%% LDA + wordembedding
# [ 用alpha調整兩者權重 alpha init=0.5 ]
#acc:  0.6465043205027494
#precision:  0.12576590730557738
#recall:  0.48930722328208587
#f1:  0.18923064905459436

#%% chanel + wordembedding
#
#acc:  0.6535742340926944
#precision:  0.12780832678711704
#recall:  0.4969719073803912
#f1:  0.19099685457455431

# [ 100epoch ]
#acc:  0.6849960722702279
#precision:  0.13864886095836607
#recall:  0.5440803501290539
#f1:  0.20853560042737992

# [ phase2-2 (CNN) ]
#acc:  0.7087912087912088
#precision:  0.14230769230769233
#recall:  0.5595481049562682
#f1:  0.21338429403370762

#%% title as gfeature 
# [ title guides text / phase 2-1 (LSTM) ]
#acc:  0.6766091051805337
#precision:  0.13744113029827318
#recall:  0.5371664050235478
#f1:  0.20648686780739606

# [ title guides text / phase 2-1 (CNN) ]
#acc:  0.6891679748822606
#precision:  0.14293563579277868
#recall:  0.5532910966584436
#f1:  0.2133955708618651

# [ co-feature guides title / phase2-2 (CNN) ]
#acc:  0.7307692307692307
#precision:  0.15070643642072212
#recall:  0.5911088061598266
#f1:  0.22545134880715267
