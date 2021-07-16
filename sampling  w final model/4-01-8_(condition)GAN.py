#CoA_group_title_condition_GAN
#----------------------------------------------------------------------------------------
#G input: data or feature / output: tag prob 
#D input: tag prob / output: real or false
#------------------------------------------------------------------------------------------

#%%
from keras.utils import Sequence
import tensorflow as tf
class DataGen(Sequence):
    def __init__(self, image_train, text_train, group_pop_train, group_cnn_train, tag_train, tot_ex, batch_size):
	    self.batch_size = batch_size
	    self.tot_ex = tot_ex #總data長度
	    self.lwr = 0
	    self.upr = batch_size//2
	    self.half = batch_size//2
	    #self.on_epoch_end()
	    self.idlist = list(range(0, tot_ex))
	    self.globalidx = np.random.choice(self.idlist, int(self.tot_ex*0.5)) #epoch global training set
	    self.G_sampleprob = np.ones(self.tot_ex) / self.tot_ex
	    self.D_sampleprob = np.ones(self.tot_ex) / self.tot_ex
	    self.image_train = image_train
	    self.text_train = text_train
	    self.group_pop_train = group_pop_train
	    self.group_cnn_train = group_cnn_train
	    self.tag_train = tag_train
    
    def __getitem__(self, index): 
        if index==0: #from global set
            general_sampleidx = self.globalidx[self.lwr:self.upr]
            #general_sampleidx = np.random.choice(self.globalidx, int(batch_size/2))
            (X,y) =  self.__data_generation(indices = general_sampleidx)
            self.lwr += self.half
            self.upr += self.half
            return (X,y)
        elif index==1: #sampling strategy for G
            G_strategic_sampleidx = np.random.choice(self.idlist, int(batch_size/2), p = self.G_sampleprob)
            (X,y) =  self.__data_generation(indices = G_strategic_sampleidx)
            return (X,y)
        else: #sampling strategy for D
            D_strategic_sampleidx = np.random.choice(self.idlist, int(batch_size/2), p = self.D_sampleprob)
            (X,y) =  self.__data_generation(indices = D_strategic_sampleidx)
            return (X,y)

    def __len__(self):
	    return int(np.floor((self.tot_ex) / self.batch_size))

    def on_epoch_end(self): #重新抽globalset＆更新sample prob.
	    self.globalidx = np.random.choice(self.idlist, int(self.tot_ex*0.5))
	    G_out, GAN_out = gan.stacked_G_D.predict([self.image_train, self.text_train, self.group_pop_train, self.group_cnn_train])
	    G_classloss = tf.keras.losses.binary_crossentropy(G_out, self.tag_train.astype(np.float32)).numpy() 
	    G_loss = G_classloss + tf.keras.losses.binary_crossentropy(GAN_out, np.ones_like(GAN_out)).numpy()  
	    if np.isnan(G_loss).any():
	        self.G_sampleprob = np.ones(self.tot_ex) / self.tot_ex
	    else:
	        self.G_sampleprob = (1/G_loss) / sum(1/G_loss)
	    if np.isnan(G_loss).any():
	        self.D_sampleprob = np.ones(self.tot_ex) / self.tot_ex
	    else:
	        self.D_sampleprob = (1/G_classloss) / sum(1/G_classloss)
	    self.lwr = 0
	    self.upr = self.batch_size//2
	    return

    def __data_generation(self, indices):
        indices.sort()
        X = [self.image_train[indices], self.text_train[indices], self.group_pop_train[indices], self.group_cnn_train[indices]]
        y = self.tag_train[indices]
        return (X,y)

datagen = DataGen(image_train, text_title_train, lda_train, lda_title_tr_arr, tag_train, tot_ex=len(image_train), batch_size=64)

#%%
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential
import random 
from numpy.random import RandomState
class GAN(object):
    def __init__(self, num_tags):
        self.num_tags = num_tags
        #self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        #self.d_optimizer = Adam(lr=0.00001, beta_1=0.5, decay=8e-8)
        #self.g_optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        #self.d_optimizer = Adam(lr=0.00001, beta_1=0.5, decay=8e-8)
        self.g_optimizer = Adam(lr=0.0008, beta_1=0.5, decay=8e-8)
        self.d_optimizer = Adam(lr=0.00001, beta_1=0.5, decay=8e-8)
        #self.g_optimizer = 'adam'

        self.G = self.__generator()
        self.G.compile(loss='BinaryCrossentropy', optimizer=self.g_optimizer)

        self.F = Model(inputs=self.G.input,
                    outputs=self.G.get_layer('co_attn_layer').output)

        self.D = self.__discriminator()
        self.D.compile(loss='BinaryCrossentropy', optimizer=self.d_optimizer)

        self.stacked_G_D = self.__stacked_generator_discriminator()
        self.stacked_G_D.compile(loss=['BinaryCrossentropy', 'BinaryCrossentropy'], loss_weights=[100,0.5], optimizer=self.g_optimizer) #, loss_weights=[50,0.5]


    def __generator(self):
        inputs_img = Input(shape=(7, 7, 512))
        inputs_text = Input(shape=(seq_length + title_length, embedding_size))
        inputs_group_pop = Input(shape=(seq_length, embedding_size)) 
        inputs_group_cnn = Input(shape=(seq_length + title_length, embedding_size))

        iFeature, iFeature_pooling = imageFeature(inputs_img)
        tFeature = textFeature(inputs_text)
        gFeature = groupFeature(inputs_group_pop) 
        eFeature = titleFeature(inputs_group_cnn)

        iFeature.set_shape((inputs_img.shape[0], num_region, embedding_size))
        tFeature.set_shape((inputs_text.shape[0], seq_length + title_length, embedding_size))
        gFeature.set_shape((inputs_group_pop.shape[0], seq_length, embedding_size))
        eFeature.set_shape((inputs_group_cnn.shape[0], seq_length + title_length, embedding_size))

        co_feature = coAttention_alt(dim_k=dim_k)([iFeature, tFeature, gFeature, eFeature])
        dropout = Dropout(drop_rate)(co_feature)
        tag_prob = Dense(num_tags, activation="sigmoid", use_bias=True)(dropout)
        model = Model(inputs=[inputs_img, inputs_text, inputs_group_pop, inputs_group_cnn], outputs=[tag_prob])
        #model.summary()
        return model 

    def __discriminator(self):
        disc_in = Input(shape=(num_tags+200,))
        x = Dense(num_tags+200, activation = LeakyReLU(alpha=.2))(disc_in) 
        #x = Dense(256, activation = LeakyReLU(alpha=.2))(x)
        x = Dense(64, activation = LeakyReLU(alpha=.2))(x)
        x = Dropout(drop_rate)(x)
        disc_out = Dense(1, activation = 'sigmoid', name = "Discriminator")(x)
        model = Model(inputs=[disc_in], outputs=[disc_out]) 
        #model.summary()
        return model

    def __stacked_generator_discriminator(self):
        self.D.trainable = False
        gan_input = self.G.input
        gen_output = self.G(gan_input)
        gen_feature = self.F(gan_input)
        gan_output = self.D(K.concatenate([gen_feature,gen_output],axis=-1))
        model = Model(inputs=gan_input, outputs=[gen_output, gan_output])
        model.summary()
        return model

    def train(self, image_train, text_train, group_pop_train, group_cnn_train, tag_train, epochs, batch_size):
        dlist, glist = [], []
        global datagen
        iters = datagen.__len__()
        for e in range(epochs):
            for cnt in range(iters):
                ## train discriminator ##  
                #fetch a batch of data to train on 
                (real_x, real_tags) = datagen.__getitem__(index=0)
                #now that we have our real example we must use self.G.predict to get our 'fake' examples	
                (tobefake_x, _) = datagen.__getitem__(index=2)
                fake_tags = self.G.predict(x=tobefake_x)
                intermediate_output = self.F(real_x)
                real_tags_temp = real_tags
                #adjust real_tags from 0/1 to (0-0.2)/(0.4-0.6)
                #real_tags = (real_tags*0.4) + np.random.uniform(0,0.2,size=real_tags.shape)
                #real_tags = (real_tags*0.4) + np.random.random(real_tags.shape)*0.2
                #real_tags = (real_tags*0.7) + np.random.random(real_tags.shape)*0.3
                #real_tags = (real_tags*0.2) + np.random.random(real_tags.shape)*0.001
                #real_tags = (real_tags*0.01*np.random.randint(81,size=real_tags.shape)) + np.random.random(real_tags.shape)*0.001
                real_tags = (real_tags*0.001*np.random.randint(250,1000,size=real_tags.shape)) + np.random.random(real_tags.shape)*0.001

				#now that we have our real and fake labels we can train discriminator 
                x_combined_batch = np.append(np.append(intermediate_output,real_tags,axis=1), np.append(intermediate_output,fake_tags,axis=1), axis=0)
                y_combined_batch = np.concatenate((np.ones((int(batch_size/2), 1)), np.zeros((int(batch_size/2), 1)))) 
                    
                #把真跟假打散
                seed = random.randint(0,10000)
                p = RandomState(seed)
                p.shuffle(x_combined_batch)
                p = RandomState(seed)
                p.shuffle(y_combined_batch)
                self.D.trainable = True
                if cnt % 2 == 0:
                    d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

                ## train generator ## 
                (tobegen_x, tobegen_y) = datagen.__getitem__(index=1)
                x_gen_batch = [np.append(real_x[0], tobegen_x[0],axis=0), np.append(real_x[1], tobegen_x[1],axis=0)
                                , np.append(real_x[2], tobegen_x[2],axis=0), np.append(real_x[3], tobegen_x[3],axis=0)]
                y_mislabled = np.ones((batch_size, 1))
                g_loss = self.stacked_G_D.train_on_batch(x_gen_batch, [np.append(real_tags, tobegen_y, axis=0), y_mislabled])
                if cnt % 10 == 0:
                    print('iter: %d, [d_loss: %f], [g_loss: %f]' % (cnt+1, d_loss, g_loss[0]))
                    
            acc_tr, _, _, _, _, _ = evaluation(tag_train, self.G.predict(x=[image_train, text_title_train, lda_train, lda_title_tr_arr]), TopK) 
            acc_te, _, _, _, _, _ = evaluation(tag_test, self.G.predict(x=[image_test, text_title_test, lda_test, lda_title_te_arr]), TopK) 
            print('epoch: %d, [d_loss: %f], [g_loss: %f], [trainacc: %f], [testacc: %f]' % (e+1, d_loss, g_loss[0], acc_tr, acc_te)) 
            print('[class_loss: %f], [adv_loss: %f]' % (g_loss[1], g_loss[2])) 
            dlist.append(d_loss)#(g_loss[1]) 
            glist.append(g_loss[0])#g_loss[2])
            datagen.on_epoch_end()
        return dlist, glist

#%%
TopK = 5
epochs = 100
batch_size = 64
round = 10

method = 'CoA_group_title_condition_GAN'
wb = load_workbook(resultdir+'evaluation2.xlsx')
ws = wb.create_sheet(title=method)
ws.append(['acc', 'precision', 'recall', 'f1', 'ndcg', 'map'])
if not os.path.exists(resultdir+method):
    os.makedirs(resultdir+method)

for i in range(round):
    print('round', i+1)
    datagen = DataGen(image_train, text_title_train, lda_train, lda_title_tr_arr, tag_train, tot_ex=len(image_train), batch_size=64)
    gan = GAN(num_tags)
    d_history, g_history = gan.train(image_train, text_title_train, lda_train, lda_title_tr_arr, tag_train.astype(np.float32), epochs=epochs, batch_size=batch_size)

    y_pred = gan.G.predict(x=[image_test, text_title_test, lda_test, lda_title_te_arr])
    acc_K, precision_K, recall_K, f1_K, ndcg_K, map_K = evaluation(tag_test, y_pred, TopK)
    print('acc: ', acc_K)
    print('precision: ', precision_K)
    print('recall: ', recall_K)
    print('f1: ', f1_K)
    print('ndcg: ', ndcg_K)
    print('map: ', map_K)
    np.savetxt(resultdir+method+'/y_pred'+str(i+1)+'(2).csv', y_pred, delimiter=",")
    save_result(acc_K, precision_K, recall_K, f1_K, ndcg_K, map_K)
wb.save(filename = resultdir+'evaluation2.xlsx')

import matplotlib.pyplot as plt
plt.title('GAN loss')
plt.xlabel('epoch')
plt.plot(range(epochs), d_history, label='D')
plt.plot(range(epochs), g_history, label='G')
plt.legend()
plt.show()

#%% observing D's prediction to  real and fake prob
(real_x, real_tags) = datagen.__getitem__(index=0)
#real_tags = (real_tags*0.2) + np.random.random(real_tags.shape)*0.001
#real_tags = (real_tags*0.9) + np.random.random(real_tags.shape)*0.1
f_real_tags = (real_tags*0.01*np.random.randint(81,size=real_tags.shape)) + np.random.random(real_tags.shape)*0.001
fake_tags = gan.G.predict(real_x)
gan_out_real = gan.D.predict(f_real_tags)
gan_out_fake = gan.D.predict(gan.G.predict(real_x))

#%%
def evaluation(y_true, y_pred, top_K):
    acc_count = 0
    precision_K = []
    recall_K = []
    f1_K = []
    ndcg_K = []
    true_sum = 0

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:] #每篇文章排名前10可能的tag index
        true_num = np.sum(y_true[i, :])
        true_sum += true_num
        dcg = 0
        idcg = 0
        idcgCount = true_num
        j = 0
        for item in top_indices:
            if y_true[i, item] == 1:
                dcg += 1.0/math.log2(j + 2)
            if idcgCount > 0:
                idcg += 1.0/math.log2(j + 2)
                idcgCount = idcgCount-1
            j += 1
        if(idcg != 0):
            ndcg_K.append(dcg/idcg)

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

    return acc_K, np.mean(np.array(precision_K)), np.mean(np.array(recall_K)), np.mean(np.array(f1_K)), np.mean(np.array(ndcg_K))

#%%GAN sampling strategy 8
#almost equal to strategy 3, but sampling prob updated by 1/G loss
#3是訓練差的(loss大的)抽樣機率提高，8是訓練好的(loss小的)抽樣機率提高
#to_be_fake sample from training set using D sample prob(updated by G's class loss)
#to_be_gen sample from training set using G sample prob(updated by G's total loss)
#D input: datagen global batchsize/2 + G(to_be_fake) batchsize/2
#G input: datagen global batchsize/2 + to_be_gen batchsize/2
#20epochs/batchsize=64
#G optimizer: adam 0.002 / D optimizer: adam 0.00002
#D 5 step per G step
#G last layer: softmax
#myLossFunc
#acc:  0.7760141093474426
#precision:  0.15343915343915346
#recall:  0.664041110271269
#f1:  0.23554040198060375




