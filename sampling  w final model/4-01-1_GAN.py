#----------------------------------------------------------------------------------------
#G input: data or feature / output: tag prob 
#D input: tag prob / output: real or false
#------------------------------------------------------------------------------------------

#%%
batch_size = 64
from keras.utils import Sequence
class DataGen(Sequence):
    def __init__(self, image_train, text_train, group_pop_train, group_cnn_train, tag_train, tot_ex, batch_size):
	    self.batch_size = batch_size
	    self.tot_ex = tot_ex
	    self.on_epoch_end()
	    self.lwr = 0
	    self.upr = batch_size//2
	    self.half = batch_size//2
	    idlist = list(range(0, tot_ex))
	    np.random.shuffle(idlist)
	    self.master = idlist
	    self.image_train = image_train
	    self.text_train = text_train
	    self.group_pop_train = group_pop_train
	    self.group_cnn_train = group_cnn_train
	    self.tag_train = tag_train
    '''
    def __getitem__(self, index):
        if index==0:
            indices = self.master[self.lwr:self.upr] #batchsize/2
            #half indices will be actual, other half will be fake
            (X,y) =  self.__data_generation(indices = indices)
            self.lwr += self.half
            self.upr += self.half
            return (X,y)
        else:
            indices = self.master[self.lwr-self.half:self.upr] #batchsize
            (X,y) =  self.__data_generation(indices = indices)
            return (X,y)
    '''
    def __getitem__(self, index):
        indices = self.master[self.lwr:self.upr] #batchsize/2
        #half indices will be actual, other half will be fake
        (X,y) =  self.__data_generation(indices = indices)
        self.lwr += self.half
        self.upr += self.half
        return (X,y)

    def __len__(self):
	    return int(np.floor((self.tot_ex) / self.batch_size))

    def on_epoch_end(self): #回到原點
	    #global lwr, upr, batchSize
		#lwr += self.batch_size
		#print lwr
		#upr += self.batch_siz
	    idlist = list(range(0, self.tot_ex))
	    np.random.shuffle(idlist) 
	    self.master = idlist 
	    self.lwr = 0
	    self.upr = self.batch_size//2
	    return

    def __data_generation(self, indices):
        indices.sort()
        X = [self.image_train[indices], self.text_train[indices], self.group_pop_train[indices], self.group_cnn_train[indices]]
        y = self.tag_train[indices]
        return (X,y)

datagen = DataGen(image_train, text_title_train, lda_train, lda_title_tr_arr, tag_train, tot_ex=len(image_train), batch_size=batch_size)

#%%
from keras.layers import LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
import random 
from numpy.random import RandomState
class GAN(object):
    def __init__(self, num_tags):
        self.num_tags = num_tags
        #self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
        self.d_optimizer = Adam(lr=0.00001, beta_1=0.5, decay=8e-8)
        self.g_optimizer = Adam(lr=0.0008, beta_1=0.5, decay=8e-8)
        #self.optimizer = 'adam'

        self.G = self.__generator()
        self.G.compile(loss='BinaryCrossentropy', optimizer=self.g_optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='BinaryCrossentropy', optimizer=self.d_optimizer)

        self.stacked_G_D = self.__stacked_generator_discriminator()
        self.stacked_G_D.compile(loss=['BinaryCrossentropy', 'BinaryCrossentropy'], loss_weights=[100,0.5], optimizer=self.g_optimizer) #loss_weights = [0.8,0.2],


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
        #co_feature = BatchNormalization()(co_feature)
        dropout = Dropout(drop_rate)(co_feature)
        tag_prob = Dense(num_tags, activation="sigmoid", use_bias=True)(dropout)
        model = Model(inputs=[inputs_img, inputs_text, inputs_group_pop, inputs_group_cnn], outputs=[tag_prob])
        #model.summary()
        return model 

    def __discriminator(self):
        disc_in = Input(shape=(num_tags,))
        #x = K.reshape(disc_in, [-1,1,num_tags])
        #x = LSTM(units=num_tags)(x)
        #x = Dense(num_tags, activation = LeakyReLU(alpha=.2))(disc_in) 
        #x = BatchNormalization()(disc_in)
        x = Dense(num_tags, activation = LeakyReLU(alpha=.2))(disc_in) 
        #x = Dropout(drop_rate)(x)
        #x = Dense(128, activation = LeakyReLU(alpha=.2))(x)
        #x = Dropout(drop_rate)(x)
        x = Dense(64, activation = LeakyReLU(alpha=.2))(x)
        #x = Dropout(drop_rate)(x)
        #x = Dense(16, activation = LeakyReLU(alpha=.2))(x)
        #x = Dropout(drop_rate)(x)
        disc_out = Dense(1, activation = 'softmax', name = "Discriminator")(x)
        model = Model(inputs=[disc_in], outputs=[disc_out]) 
        #model.summary()
        return model

    def __stacked_generator_discriminator(self):
        self.D.trainable = False
        gan_input = self.G.input
        gen_output = self.G(gan_input)
        gan_output = self.D(gen_output)
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
                fake_tags = self.G.predict(x=real_x)
                #adjust real_tags from 0/1 to (0-0.2)/(0.4-0.6)
                #real_tags = (real_tags*0.4) + np.random.uniform(0,0.2,size=real_tags.shape)
                #real_tags = (real_tags*0.4) + np.random.random(real_tags.shape)*0.2
                #real_tags = (real_tags*0.9) + np.random.random(real_tags.shape)*0.1
                #real_tags = (real_tags*0.2) + np.random.random(real_tags.shape)*0.001
                real_tags = (real_tags*0.001*np.random.randint(250,1000,size=real_tags.shape)) + np.random.random(real_tags.shape)*0.001

				#now that we have our real and fake labels we can train discriminator 
                x_combined_batch = np.append(real_tags, fake_tags, axis=0)
                y_combined_batch = np.append(np.ones((int(batch_size/2), 1)), np.zeros((int(batch_size/2), 1)), axis=0) 
                    
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
                y_mislabled = np.ones((batch_size//2, 1))
                g_loss = self.stacked_G_D.train_on_batch(real_x, [real_tags, y_mislabled])
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

    '''
    def train(self, image_train, text_train, tag_train, epochs, batch_size=128):
    
        for cnt in range(epochs):
            ## train discriminator
            #抽batchsize/2的作為real data(todo:寫成data generation func)
            random_index = np.random.randint(0, len(tag_train)-batch_size/2)
            real_tag = tag_train[random_index : random_index+int(batch_size/2)]

            gen_index = np.random.randint(0, len(tag_train)-batch_size/2) 
            gen_tag = self.G.predict(x=[image_train[gen_index : gen_index+int(batch_size/2)], text_train[gen_index : gen_index+int(batch_size/2)]])

            x_combined_batch = np.concatenate((real_tag, gen_tag))
            y_combined_batch = np.concatenate((np.ones((int(batch_size/2), 1)), np.zeros((int(batch_size/2), 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

            ## train generator
            sample_idx = random.sample(range(len(image_train)),batch_size)
            y_mislabled = np.ones((batch_size, 1))

            g_loss = self.stacked_G_D.train_on_batch([image_train[sample_idx], text_train[sample_idx]], y_mislabled)
            
            if cnt % 10 == 0:
                print('epoch: %d, [d_loss: %f], [g_loss: %f]' % (cnt, d_loss[0], g_loss))
    '''
#%%
TopK = 5
epochs = 100
gan = GAN(num_tags)
text_title_train = np.concatenate((embedding_train, embedding_title_train), axis=1)
text_title_test = np.concatenate((embedding_test, embedding_title_test), axis=1)
d_history, g_history = gan.train(image_train, text_title_train, lda_train, lda_title_tr_arr, tag_train.astype(np.float32), epochs=epochs, batch_size=batch_size)

#%%
y_pred = gan.G.predict(x=[image_test, text_title_test, lda_test, lda_title_te_arr])
acc_K, precision_K, recall_K, f1_K, ndcg_K, map_K = evaluation(tag_test, y_pred, TopK)
print('acc: ', acc_K)
print('precision: ', precision_K)
print('recall: ', recall_K)
print('f1: ', f1_K)
print('ndcg: ', ndcg_K)
print('map: ', map_K)

import matplotlib.pyplot as plt
plt.title('GAN loss')
plt.xlabel('epoch')
plt.plot(range(epochs), d_history, label='D')
plt.plot(range(epochs), g_history, label='G')
plt.legend()
plt.show()

#%%
gan_out_test = gan.D.predict(y_pred)
mislabled_test = np.ones((len(tag_test), 1))
loss_test = myLossFunc(mislabled_test, gan_out_test)

#%% observing D's prediction to  real and fake prob
(real_x, real_tags) = datagen.__getitem__(index=0)
real_tags = (real_tags*0.2) + np.random.random(real_tags.shape)*0.001
gan_out_real = gan.D.predict(real_tags)
gan_out_fake = gan.D.predict(gan.G.predict(real_x))

#%% GAN/100 epochs
#acc:  0.32128829536527886
#precision:  0.042262372348782405
#recall:  0.1515299442636442
#f1:  0.06281316145462905

#%% GAN/300 epochs
#acc:  0.06441476826394343
#precision:  0.006598586017282011
#recall:  0.030729809598623425
#f1:  0.010127987549317432

#%% GAN/20epochs/batchsize=128
#G input: datagen batchsize/2 + random batchsize/2
#D input: datagen=> real batchsize/2 +　gen batchsize/2
#D train per 5 steps
#acc:  0.03613511390416339
#precision:  0.007541241162608014
#recall:  0.01798339131410616
#f1:  0.009812523166804393

#%%GAN sampling strategy 1
#D input: datagen real batchsize/2 + G(real) batchsize/2
#G input: datagen real batchsize/2
#20epochs/batchsize=64
#G optimizer: adam 0.002 / D optimizer: adam 0.00002
#D 5 step per G step
#G last layer: softmax
#myLossFunc
#acc:  0.7566137566137566
#precision:  0.14955908289241623
#recall:  0.6401444528428655
#f1:  0.22909482740555526
